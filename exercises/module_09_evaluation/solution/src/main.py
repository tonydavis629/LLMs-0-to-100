"""
Module 9 Exercise runner: build a small benchmark suite and score two models

Run with:
    uv run python module_09_evaluation/src/main.py

Loads two finished checkpoints — the Module 6 instruct model and the Module 7 GRPO
model — and runs the same evaluation suite over both: perplexity on held-out text,
exact match and token F1 on 50 generated answers, likelihood-scored multiple choice,
and pass@k over sampled completions. It prints the protocol first, then the per-task
table, then the comparison, and saves a grouped bar chart.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one metric at a time and re-run immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    perplexity,
    normalize_answer,
    exact_match,
    token_f1,
    score_multiple_choice,
    pass_at_k,
    task_accuracy,
    suite_score,
)
from model import load_instruct_model, generate  # noqa: E402
from tokenizer import encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import load_jsonl  # noqa: E402
from visualization import save_task_comparison  # noqa: E402


# ---------------------------------------------------------------------------
# The protocol. Every one of these knobs moves the scores, which is why a
# reproducible report prints them before it prints any number.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128         # context window used for the loss and for generation
MAX_NEW_TOKENS = 14      # generation budget per case (longest answer is 12 chars)
N_SAMPLES = 5            # sampled completions per case, for pass@k
TEMPERATURE = 0.8        # sampling temperature for those completions
PASS_K = 5               # the k in the reported pass@k
SEED = 1337

TASK_ORDER = ["uppercase", "repeat", "reverse", "qa"]
# Both checkpoints ship with the repo and were pretrained for this module by
# solution/src/make_checkpoints.py: instruct_model.pt is the Module 5 base model after
# multi-task SFT (the Module 6 story), and rl_model.pt is that same checkpoint after GRPO
# on `reverse` only (the Module 7 story).
MODELS = [("instruct", "instruct_model.pt"), ("rl", "rl_model.pt")]

_THIS_DIR = Path(__file__).resolve().parent
END_TOKEN = "<|end|>"


def _find_data_file(name: str) -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _is_implemented(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception still means the student wrote *something*.
        return True


def _probe_steps() -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs."""
    return {
        "perplexity": _is_implemented(perplexity, 1.5),
        "normalize_answer": _is_implemented(normalize_answer, " Blue. "),
        "exact_match": _is_implemented(exact_match, "blue", ["blue"]),
        "token_f1": _is_implemented(token_f1, "it is blue", "it is blue"),
        "score_multiple_choice": _is_implemented(score_multiple_choice, [-1.0, -2.0], [2, 2]),
        "pass_at_k": _is_implemented(pass_at_k, 5, 2, 3),
        "task_accuracy": _is_implemented(task_accuracy, {"qa": [1.0, 0.0]}),
        "suite_score": _is_implemented(suite_score, {"qa": 0.5}),
    }


def _heading(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Running the models (PROVIDED). None of this is a metric; it is the plumbing
# that turns a checkpoint plus a case into the raw material a metric consumes.
# ---------------------------------------------------------------------------


@torch.no_grad()
def mean_token_loss(model, text: str, stoi: dict[str, int]) -> float:
    """Average next-token cross-entropy over held-out text, in nats per token.

    The text is cut into non-overlapping BLOCK_SIZE windows and scored the same way
    Module 5 measured validation loss. This is the input to perplexity, and the one
    measurement in the suite that needs no labels and no generation.
    """
    ids = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(ids) - BLOCK_SIZE - 1, BLOCK_SIZE):
        window = ids[start:start + BLOCK_SIZE + 1]
        logits = model(window[:-1].unsqueeze(0))[0]
        loss = F.cross_entropy(logits, window[1:], reduction="sum")
        total_loss += loss.item()
        total_tokens += window.shape[0] - 1
    return total_loss / total_tokens


def _prefix_ids(prompt: str, special: dict[str, int], enc) -> torch.Tensor:
    """The chat-template generation prefix: [user] prompt [end] [assistant]."""
    ids = ([special["<|user|>"]] + enc(prompt) + [special["<|end|>"]]
           + [special["<|assistant|>"]])
    return torch.tensor([ids], dtype=torch.long)


def _response_text(seq: torch.Tensor, prompt_len: int, itos) -> str:
    """Decode the generated portion and cut it at the first <|end|>."""
    return decode(seq[prompt_len:], itos).split(END_TOKEN, 1)[0]


def run_cases(model, cases, special, enc, itos) -> list[dict]:
    """Generate one greedy answer and N_SAMPLES sampled answers per case.

    Two decoding protocols on the same model in one pass: the greedy answer is what
    exact match and F1 score, and the sampled answers are what pass@k needs. The
    sampler is seeded per case so the run is reproducible.
    """
    out = []
    for case in cases:
        prefix = _prefix_ids(case["prompt"], special, enc)
        prompt_len = prefix.shape[1]
        greedy = generate(model, prefix, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
        gen = torch.Generator().manual_seed(SEED)
        samples = [
            _response_text(
                generate(model, prefix, MAX_NEW_TOKENS, BLOCK_SIZE,
                         temperature=TEMPERATURE, generator=gen)[0],
                prompt_len, itos)
            for _ in range(N_SAMPLES)
        ]
        out.append({
            "id": case["id"],
            "task": case["task"],
            "prompt": case["prompt"],
            "answers": case["answers"],
            "greedy": _response_text(greedy[0], prompt_len, itos),
            "samples": samples,
        })
    return out


@torch.no_grad()
def option_log_probs(model, question: str, options: list[str], special, enc):
    """Total log-probability and token count for each multiple-choice option.

    No generation happens here. Each option is appended to the prompt and scored
    under the model, exactly the way MMLU and HellaSwag are run on base models. The
    two lists returned are what `score_multiple_choice` compares.
    """
    totals, lengths = [], []
    for option in options:
        prefix = _prefix_ids(question, special, enc)[0].tolist()
        option_ids = enc(option)
        seq = torch.tensor([prefix + option_ids], dtype=torch.long)
        logits = model(seq[:, :-1])[0]
        log_probs = F.log_softmax(logits, dim=-1)
        # Score only the option's own tokens: positions len(prefix)-1 .. end predict them.
        scored = log_probs[len(prefix) - 1:].gather(
            -1, torch.tensor(option_ids).unsqueeze(-1)).squeeze(-1)
        totals.append(scored.sum().item())
        lengths.append(len(option_ids))
    return totals, lengths


# ---------------------------------------------------------------------------
# Scoring (this is where the functions you wrote get used)
# ---------------------------------------------------------------------------


def score_records(records: list[dict], steps: dict[str, bool]) -> dict[str, dict[str, list[float]]]:
    """Turn raw generations into per-task score lists for each metric."""
    out: dict[str, dict[str, list[float]]] = {
        metric: {task: [] for task in TASK_ORDER}
        for metric in ("em", "f1", "pass1", "passk")
    }
    for rec in records:
        task = rec["task"]
        if steps["exact_match"]:
            out["em"][task].append(exact_match(rec["greedy"], rec["answers"]))
        if steps["token_f1"]:
            out["f1"][task].append(max(token_f1(rec["greedy"], a) for a in rec["answers"]))
        if steps["pass_at_k"] and steps["exact_match"]:
            correct = sum(int(exact_match(s, rec["answers"])) for s in rec["samples"])
            out["pass1"][task].append(pass_at_k(N_SAMPLES, correct, 1))
            out["passk"][task].append(pass_at_k(N_SAMPLES, correct, PASS_K))
    return out


def _shorten(text: str, limit: int = 22) -> str:
    """Trim a generation for display. A model that never emits <|end|> runs long."""
    return text if len(text) <= limit else text[:limit] + "..."


def _pct(value: float | None) -> str:
    return "   --" if value is None else f"{value:6.1%}"


def _print_metric_table(title: str, note: str, per_task: dict[str, dict[str, float]],
                        counts: dict[str, int]) -> None:
    """Print one metric's per-task numbers for both models, plus the difference."""
    print(f"  {title}")
    print(f"  {note}")
    print(f"    {'task':<12}{'cases':>7}{'instruct':>12}{'rl':>10}{'diff':>10}")
    for task in TASK_ORDER:
        left = per_task["instruct"].get(task)
        right = per_task["rl"].get(task)
        diff = "" if left is None or right is None else f"{right - left:+9.1%}"
        print(f"    {task:<12}{counts[task]:>7}{_pct(left):>12}{_pct(right):>10}{diff:>10}")
    print()


def print_report(scored: dict[str, dict], counts: dict[str, int], steps: dict[str, bool]) -> None:
    """The per-task tables. Averages come last, and never alone."""
    metrics = [
        ("em", "EXACT MATCH (greedy decoding)",
         "The strictest metric: the normalized answer must equal an acceptable answer."),
        ("f1", "TOKEN F1 (greedy decoding)",
         "Partial credit for overlapping tokens; equals exact match on one-word answers."),
        ("pass1", f"pass@1 (sampled at temperature {TEMPERATURE})",
         "Accuracy of a single sampled answer, estimated from N=5 samples per case."),
        ("passk", f"pass@{PASS_K} (sampled at temperature {TEMPERATURE})",
         "Is the right answer anywhere in the model's distribution across 5 tries?"),
    ]
    if not steps["task_accuracy"]:
        scored_metrics = [t for k, t, _ in metrics if scored["instruct"][k]["reverse"]]
        print("  [per-task tables need task_accuracy() from step 5]")
        print(f"  Scored and waiting: {', '.join(scored_metrics) or 'nothing yet'}")
        print()
        return

    for key, title, note in metrics:
        per_task = {}
        for name in ("instruct", "rl"):
            lists = scored[name][key]
            usable = {t: v for t, v in lists.items() if v}
            per_task[name] = task_accuracy(usable) if usable else {}
        if any(per_task.values()):
            _print_metric_table(title, note, per_task, counts)


# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)

    models = {}
    for name, filename in MODELS:
        model, stoi, itos = load_instruct_model(_find_data_file(filename))
        models[name] = model
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    enc = lambda s: encode(s, stoi)  # noqa: E731

    cases = load_jsonl(_find_data_file("tasks.jsonl"))
    mc_cases = load_jsonl(_find_data_file("multiple_choice.jsonl"))
    held_out = _find_data_file("held_out.txt").read_text(encoding="utf-8")
    counts = {task: sum(1 for c in cases if c["task"] == task) for task in TASK_ORDER}

    steps = _probe_steps()

    # ------------------------------------------------------------------
    # The protocol comes first. A benchmark number belongs to a model AND a
    # protocol, so a report that does not state one cannot be reproduced.
    # ------------------------------------------------------------------
    _heading("MODULE 9: evaluating two finished checkpoints")
    print("Models under test")
    print(f"  instruct  data/instruct_model.pt   Module 6: multi-task SFT")
    print(f"  rl        data/rl_model.pt         Module 7: GRPO on `reverse` only")
    print(f"  Both are the same TinyGPT architecture, tokenizer, and chat template.")
    print()
    print("Protocol")
    print(f"  chat template     <|user|> PROMPT <|end|> <|assistant|> ANSWER <|end|>")
    print(f"  normalization     lowercase, strip punctuation, collapse whitespace")
    print(f"  generation budget {MAX_NEW_TOKENS} tokens    context {BLOCK_SIZE}")
    print(f"  decoding          greedy for exact match and F1;"
          f" {N_SAMPLES} samples at temperature {TEMPERATURE} for pass@k")
    print(f"  seed              {SEED} (per case, so both models see the same draws)")
    print(f"  suite             {len(cases)} generated cases across {len(TASK_ORDER)} tasks,"
          f" {len(mc_cases)} multiple-choice questions")
    print(f"  held-out text     {len(held_out):,} characters, never seen in training")
    print()

    # ------------------------------------------------------------------
    # 1. Perplexity: no labels, no generation, no instruction-following needed.
    # ------------------------------------------------------------------
    _heading("1. PERPLEXITY ON HELD-OUT TEXT")
    if steps["perplexity"]:
        print(f"    {'model':<12}{'loss (nats)':>14}{'perplexity':>14}")
        for name in models:
            loss = mean_token_loss(models[name], held_out, stoi)
            print(f"    {name:<12}{loss:>14.4f}{perplexity(loss):>14.2f}")
        print("  Lower is better. This says nothing about whether either model")
        print("  follows instructions, which is exactly its limitation.")
    else:
        print("  [skipped: implement perplexity()]")
    print()

    # ------------------------------------------------------------------
    # 2. Generation, then the generated-answer metrics.
    # ------------------------------------------------------------------
    _heading("2. TASK SUITE")
    gen_ready = steps["normalize_answer"] and (steps["exact_match"] or steps["token_f1"])
    scored: dict[str, dict] = {}
    per_task_em: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    if not gen_ready:
        print("  [skipped: implement normalize_answer() and exact_match()]")
        print()
    else:
        print(f"  Generating {len(cases)} greedy + {len(cases) * N_SAMPLES} sampled"
              f" answers per model...")
        print()
        all_records = {}
        for name, model in models.items():
            all_records[name] = run_cases(model, cases, special, enc, itos)
            scored[name] = score_records(all_records[name], steps)
        print_report(scored, counts, steps)

        # One case per task, side by side. Averages are easier to argue with when
        # the failures they summarize are on the same page.
        print("  Sample cases (greedy), one per task:")
        for task in TASK_ORDER:
            pairs = [(a, b) for a, b in zip(all_records["instruct"], all_records["rl"])
                     if a["task"] == task]
            if not pairs:
                continue
            # Prefer a case where the two models disagree; fall back to the first.
            left, right = next(
                ((a, b) for a, b in pairs if a["greedy"] != b["greedy"]), pairs[0])
            print(f"    [{task}] {left['prompt']!r}   want {left['answers'][0]!r}")
            print(f"        instruct {_shorten(left['greedy'])!r}")
            print(f"        rl       {_shorten(right['greedy'])!r}")
        print()

    # ------------------------------------------------------------------
    # 3. Multiple choice, scored by likelihood (no generation at all).
    # ------------------------------------------------------------------
    _heading("3. MULTIPLE CHOICE, SCORED BY LIKELIHOOD")
    mc_accuracy: dict[str, float] = {}
    if steps["score_multiple_choice"]:
        for name, model in models.items():
            correct = 0
            for case in mc_cases:
                totals, lengths = option_log_probs(
                    model, case["question"], case["options"], special, enc)
                correct += int(score_multiple_choice(totals, lengths) == case["answer_index"])
            mc_accuracy[name] = correct / len(mc_cases)
            print(f"    {name:<12}{correct:>3}/{len(mc_cases)}   {mc_accuracy[name]:>6.1%}")
        print(f"  Chance is 25%. Nothing was generated: each option was scored under")
        print(f"  the model and the highest per-token log-probability won.")
    else:
        print("  [skipped: implement score_multiple_choice()]")
    print()

    # ------------------------------------------------------------------
    # 4. The headline number, and what it hides.
    # ------------------------------------------------------------------
    _heading("4. SUITE SCORE")
    if gen_ready and steps["task_accuracy"] and steps["suite_score"] and steps["exact_match"]:
        for name in models:
            per_task_em[name] = task_accuracy(scored[name]["em"])
            overall[name] = suite_score(per_task_em[name])
            print(f"    {name:<12}{overall[name]:>8.1%}   (mean of the four task scores)")
        delta = overall["rl"] - overall["instruct"]
        print()
        print(f"  Overall difference: {delta:+.1%}")
        print("  Read the per-task table above before believing that number.")
        out_img = _THIS_DIR.parent / "output" / "task_comparison.png"
        save_task_comparison(
            TASK_ORDER,
            [per_task_em["instruct"][t] for t in TASK_ORDER],
            [per_task_em["rl"][t] for t in TASK_ORDER],
            out_img,
            overall["instruct"],
            overall["rl"],
        )
        print(f"  Chart saved to {out_img}")
    else:
        print("  [skipped: implement task_accuracy() and suite_score()]")
    print()

    _heading("Done")
    print("Run after each step; unfinished steps are skipped automatically.")


if __name__ == "__main__":
    main()
