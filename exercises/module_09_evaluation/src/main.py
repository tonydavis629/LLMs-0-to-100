"""
Module 9 Exercise runner: build a small benchmark suite and score two models

Run with:
    uv run python module_09_evaluation/src/main.py

Loads two finished checkpoints (the Module 6 instruct model and the Module 7 GRPO
model) and scores both with the metrics you write in exercise.py. The protocol is
printed first. Then every step is tagged on its header line, and its output
follows: what your metric reports about the two models, then the result of each
test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-8); the protocol is printed only for a full run.
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

from exercise import (  # noqa: E402  (import after sys.path edits)
    perplexity,
    normalize_answer,
    exact_match,
    token_f1,
    task_accuracy,
    suite_score,
    score_multiple_choice,
    pass_at_k,
)
from model import load_instruct_model, generate  # noqa: E402
from tokenizer import encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import load_jsonl  # noqa: E402
from visualization import save_task_comparison  # noqa: E402

# One test file per step lives in tests/
from tests.test_step1_perplexity import check_perplexity  # noqa: E402
from tests.test_step2_normalize import check_normalize_answer  # noqa: E402
from tests.test_step3_exact_match import check_exact_match  # noqa: E402
from tests.test_step4_token_f1 import check_token_f1  # noqa: E402
from tests.test_step5_task_accuracy import check_task_accuracy  # noqa: E402
from tests.test_step6_suite_score import check_suite_score  # noqa: E402
from tests.test_step7_multiple_choice import check_score_multiple_choice  # noqa: E402
from tests.test_step8_pass_at_k import check_pass_at_k  # noqa: E402


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
# src/make_checkpoints.py: instruct_model.pt is the Module 5 base model after
# multi-task SFT (the Module 6 story), and rl_model.pt is that same checkpoint after GRPO
# on `reverse` only (the Module 7 story).
MODELS = [("instruct", "instruct_model.pt"), ("rl", "rl_model.pt")]

_THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _THIS_DIR.parent / "output"
END_TOKEN = "<|end|>"


def _find_data_file(name: str) -> Path:
    """Walk up from this file until we find data/<name>."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


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
# Loading the suite and printing the protocol (PROVIDED)
# ---------------------------------------------------------------------------


@dataclass
class Suite:
    """Everything the steps share: both models, the tokenizer maps, and the data."""

    models: dict           # "instruct" / "rl" -> a loaded TinyGPT
    stoi: dict             # character -> token id (same for both models)
    itos: dict             # token id -> character
    special: dict          # special token string -> its id
    cases: list            # the 50 generated-answer cases from tasks.jsonl
    mc_cases: list         # the 16 multiple-choice questions
    held_out: str          # held-out text for perplexity
    counts: dict           # task -> number of cases in that task

    def enc(self, text: str) -> list[int]:
        """Encode a plain string into token ids."""
        return encode(text, self.stoi)


def load_suite() -> Suite:
    """Load both checkpoints and every evaluation file."""
    models = {}
    for name, filename in MODELS:
        model, stoi, itos = load_instruct_model(_find_data_file(filename))
        models[name] = model
    cases = load_jsonl(_find_data_file("tasks.jsonl"))
    return Suite(
        models=models,
        stoi=stoi,
        itos=itos,
        special={tok: stoi[tok] for tok in SPECIAL_TOKENS},
        cases=cases,
        mc_cases=load_jsonl(_find_data_file("multiple_choice.jsonl")),
        held_out=_find_data_file("held_out.txt").read_text(encoding="utf-8"),
        counts={task: sum(1 for c in cases if c["task"] == task) for task in TASK_ORDER},
    )


def print_protocol(suite: Suite) -> None:
    """The protocol comes first. A benchmark number belongs to a model AND a
    protocol, so a report that does not state one cannot be reproduced."""
    print("MODULE 9: evaluating two finished checkpoints")
    print()
    print("Models under test")
    print("  instruct  data/instruct_model.pt   Module 6: multi-task SFT")
    print("  rl        data/rl_model.pt         Module 7: GRPO on `reverse` only")
    print("  Both are the same TinyGPT architecture, tokenizer, and chat template.")
    print()
    print("Protocol")
    print("  chat template     <|user|> PROMPT <|end|> <|assistant|> ANSWER <|end|>")
    print("  normalization     lowercase, strip punctuation, collapse whitespace")
    print(f"  generation budget {MAX_NEW_TOKENS} tokens    context {BLOCK_SIZE}")
    print("  decoding          greedy for exact match and F1")
    print(f"                    {N_SAMPLES} samples at temperature {TEMPERATURE} for pass@k")
    print(f"  seed              {SEED} (per case, so both models see the same draws)")
    print(f"  suite             {len(suite.cases)} generated cases across {len(TASK_ORDER)} tasks,"
          f" {len(suite.mc_cases)} multiple-choice questions")
    print(f"  held-out text     {len(suite.held_out):,} characters, never seen in training")
    print()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status




# ---------------------------------------------------------------------------
# Unfinished earlier steps
# ---------------------------------------------------------------------------


def _require(step: str, call, why: str) -> None:
    """Stop this step with a pointed message if an earlier step is unfinished."""
    try:
        call()
    except NotImplementedError:
        raise NotImplementedError(f"needs {step} to {why}") from None


def _probe_own(fn, *args, why: str) -> None:
    """Call exact_match() or token_f1() once on a tiny input, before the real work.

    Both call normalize_answer() (Step 2) before they reach their own blank. So
    the probe first swaps in a stand-in normalize_answer() that returns the text
    unchanged: if the function still stops with a TODO, the TODO is its own and
    that is what the step reports. Only then does it check Step 2 is finished.
    """
    names = fn.__globals__                          # the exercise module's global names
    real = names["normalize_answer"]
    names["normalize_answer"] = lambda text: text   # stand-in: text unchanged
    try:
        fn(*args)
    except NotImplementedError:
        raise                                       # the step's own blank is unfinished
    except Exception:  # noqa: BLE001
        pass                                        # other problems surface in the real run
    finally:
        names["normalize_answer"] = real            # always put the real one back
    _require("Step 2 (normalize_answer)", lambda: normalize_answer("x"), why)


# ---------------------------------------------------------------------------
# Generating and scoring answers (PROVIDED). Generation is the slow part, so it
# runs once and every step that needs the answers reuses them.
# ---------------------------------------------------------------------------


def get_records(suite: Suite, results: dict) -> dict[str, list[dict]]:
    """Both models' greedy and sampled answers for every case, generated once."""
    if "records" not in results:
        print(f"  Generating {len(suite.cases)} greedy + {len(suite.cases) * N_SAMPLES}"
              f" sampled answers per model...")
        results["records"] = {
            name: run_cases(model, suite.cases, suite.special, suite.enc, suite.itos)
            for name, model in suite.models.items()
        }
    return results["records"]


def scores_by_task(records: list[dict], score) -> dict[str, list[float]]:
    """Apply `score(record)` to every case and group the scores by task."""
    grouped: dict[str, list[float]] = {task: [] for task in TASK_ORDER}
    for rec in records:
        grouped[rec["task"]].append(score(rec))
    return grouped


def per_task_scores(records: list[dict], score) -> dict[str, float]:
    """One model's scores, grouped by task and averaged by your task_accuracy()."""
    per_task = task_accuracy(scores_by_task(records, score))
    if not isinstance(per_task, dict):
        raise TypeError("task_accuracy() should return a dict of task -> mean score,"
                        f" got a {type(per_task).__name__}")
    return per_task


def greedy_em(rec: dict) -> float:
    """Exact match of the greedy answer (your Step 3)."""
    return exact_match(rec["greedy"], rec["answers"])


def greedy_f1(rec: dict) -> float:
    """Token F1 of the greedy answer against its closest acceptable answer (your Step 4)."""
    return max(token_f1(rec["greedy"], answer) for answer in rec["answers"])


def samples_correct(rec: dict) -> int:
    """How many of the N_SAMPLES sampled answers are exactly right."""
    return sum(int(exact_match(s, rec["answers"])) for s in rec["samples"])


def sample_pair(records: dict[str, list[dict]], task: str) -> tuple[dict, dict]:
    """One case from `task`, as answered by each model.

    Prefers a case where the two models disagree, since that is the informative
    one; falls back to the first case of the task.
    """
    pairs = [(a, b) for a, b in zip(records["instruct"], records["rl"]) if a["task"] == task]
    return next(((a, b) for a, b in pairs if a["greedy"] != b["greedy"]), pairs[0])


def _shorten(text: str, limit: int = 22) -> str:
    """Trim a generation for display. A model that never emits <|end|> runs long."""
    return text if len(text) <= limit else text[:limit] + "..."


def _pct(value: float | None) -> str:
    return "   --" if value is None else f"{value:6.1%}"


def print_metric_table(title: str, note: str, per_task: dict[str, dict[str, float]],
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


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(suite: Suite) -> str:
    """Perplexity on held-out text: no labels, no generation."""

    def show():
        perplexity(1.0)  # an unfinished perplexity() stops here with its TODO
        print(f"    {'model':<12}{'loss (nats)':>14}{'perplexity':>14}")
        for name, model in suite.models.items():
            loss = mean_token_loss(model, suite.held_out, suite.stoi)
            print(f"    {name:<12}{loss:>14.4f}{perplexity(loss):>14.2f}")
        print("  Lower is better. This says nothing about whether either model")
        print("  follows instructions, which is exactly its limitation.")

    return run_step("Step 1: perplexity()", show, lambda: check_perplexity(perplexity))


def step_2() -> str:
    """The same answer, formatted four ways, and what normalization makes of it."""

    def show():
        print("  One answer formatted four ways, after normalize_answer():")
        for raw in ("It is down.", "  it is   down ", "IT IS DOWN!", "it is\tdown\n"):
            print(f"    {raw!r:<20} -> {normalize_answer(raw)!r}")

    return run_step("Step 2: normalize_answer()", show,
                    lambda: check_normalize_answer(normalize_answer))


def step_3(suite: Suite, results: dict) -> str:
    """Exact match on the greedy answers: the count, and one case per task."""

    def show():
        _probe_own(exact_match, "blue", ["blue"], why="compare answers")
        records = get_records(suite, results)
        total = len(suite.cases)
        # How many of the 50 greedy answers each model gets exactly right
        hits = {name: sum(greedy_em(rec) for rec in recs) for name, recs in records.items()}
        print(f"  Greedy answers that match exactly: instruct {hits['instruct']:.0f}/{total},"
              f" rl {hits['rl']:.0f}/{total}")
        print("  One case per task, with its exact-match score:")
        for task in TASK_ORDER:
            left, right = sample_pair(records, task)
            print(f"    [{task}] {left['prompt']!r}   want {left['answers'][0]!r}")
            for name, rec in (("instruct", left), ("rl", right)):
                shown = repr(_shorten(rec["greedy"]))
                print(f"        {name:<8} {shown:<28} {greedy_em(rec):.1f}")

    return run_step("Step 3: exact_match()", show, lambda: check_exact_match(exact_match))


def step_4(suite: Suite, results: dict) -> str:
    """Token F1 on the greedy answers: where does it give partial credit?"""

    def show():
        _probe_own(token_f1, "it is blue", "it is blue", why="split answers into tokens")
        records = get_records(suite, results)
        # Every (model, case, F1) where F1 is neither a full miss nor a full hit
        partial = [(name, rec, greedy_f1(rec))
                   for name, recs in records.items() for rec in recs
                   if 0.0 < greedy_f1(rec) < 1.0]
        print("  Greedy answers that earn partial credit (F1 between 0 and 1):")
        for name, rec, f1 in partial[:3]:  # a few examples are enough
            print(f"    [{rec['task']}] {rec['prompt']!r}   want {rec['answers'][0]!r}")
            shown = repr(_shorten(rec["greedy"]))
            print(f"        {name:<8} {shown:<28} {f1:.2f}")
        if len(partial) > 3:
            print(f"    ... and {len(partial) - 3} more")
        if not partial:
            print("    none: every greedy answer scores exactly 0 or 1")

    return run_step("Step 4: token_f1()", show, lambda: check_token_f1(token_f1))


def step_5(suite: Suite, results: dict) -> str:
    """The per-task tables for exact match and F1."""

    def show():
        task_accuracy({"qa": [1.0, 0.0]})  # an unfinished task_accuracy() stops here
        _require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                 "score the answers it averages")
        _require("Step 4 (token_f1)", lambda: token_f1("blue", "blue"),
                 "score the answers it averages")
        records = get_records(suite, results)
        tables = [
            (greedy_em, "EXACT MATCH (greedy decoding)",
             "The strictest metric: the normalized answer must equal an acceptable answer."),
            (greedy_f1, "TOKEN F1 (greedy decoding)",
             "Partial credit for overlapping tokens; equals exact match on one-word answers."),
        ]
        for i, (score, title, note) in enumerate(tables):
            if i > 0:
                print()
            per_task = {name: per_task_scores(recs, score) for name, recs in records.items()}
            print_metric_table(title, note, per_task, suite.counts)

    return run_step("Step 5: task_accuracy()", show, lambda: check_task_accuracy(task_accuracy))


def step_6(suite: Suite, results: dict) -> str:
    """The headline number, what it hides, and the bar chart."""

    def show():
        suite_score({"qa": 0.5})  # an unfinished suite_score() stops here with its TODO
        _require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                 "score the answers")
        _require("Step 5 (task_accuracy)", lambda: task_accuracy({"qa": [1.0]}),
                 "average the scores within each task")
        records = get_records(suite, results)
        per_task_em, overall, pooled = {}, {}, {}
        for name, recs in records.items():
            per_task_em[name] = per_task_scores(recs, greedy_em)
            overall[name] = suite_score(per_task_em[name])
            # The other protocol: one average over all 50 cases, ignoring tasks
            pooled[name] = sum(greedy_em(rec) for rec in recs) / len(recs)
            print(f"    {name:<12}{overall[name]:>8.1%}   (mean of the four task scores)")
        print(f"  Overall difference: {overall['rl'] - overall['instruct']:+.1%}")
        print(f"  Averaging all {len(suite.cases)} cases instead gives instruct {pooled['instruct']:.1%},"
              f" rl {pooled['rl']:.1%} ({pooled['rl'] - pooled['instruct']:+.1%}).")
        print("  Read the per-task table before believing either number.")

        save_task_comparison(
            TASK_ORDER,
            [per_task_em["instruct"][t] for t in TASK_ORDER],
            [per_task_em["rl"][t] for t in TASK_ORDER],
            OUTPUT_DIR / "task_comparison.png",
            overall["instruct"],
            overall["rl"],
        )
        print("  Saved chart to output/task_comparison.png")

    return run_step("Step 6: suite_score()", show, lambda: check_suite_score(suite_score))


def step_7(suite: Suite) -> str:
    """Multiple choice, scored by likelihood: nothing is generated."""

    def show():
        score_multiple_choice([-1.0, -2.0], [2, 2])  # an unfinished function stops here
        for name, model in suite.models.items():
            correct = 0
            for case in suite.mc_cases:
                totals, lengths = option_log_probs(
                    model, case["question"], case["options"], suite.special, suite.enc)
                correct += int(score_multiple_choice(totals, lengths) == case["answer_index"])
            accuracy = correct / len(suite.mc_cases)
            print(f"    {name:<12}{correct:>3}/{len(suite.mc_cases)}   {accuracy:>6.1%}")
        print("  Chance is 25%. Nothing was generated: each option was scored under")
        print("  the model and the highest per-token log-probability won.")

    return run_step("Step 7: score_multiple_choice()", show,
                    lambda: check_score_multiple_choice(score_multiple_choice))


def step_8(suite: Suite, results: dict) -> str:
    """pass@1 and pass@5 from the sampled answers, per task."""

    def show():
        pass_at_k(5, 2, 3)  # an unfinished pass_at_k() stops here with its TODO
        _require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                 "check each sampled answer")
        _require("Step 5 (task_accuracy)", lambda: task_accuracy({"qa": [1.0]}),
                 "average the estimates within each task")
        records = get_records(suite, results)
        tables = [
            (1, f"pass@1 (sampled at temperature {TEMPERATURE})",
             f"Accuracy of a single sampled answer, estimated from N={N_SAMPLES} samples per case."),
            (PASS_K, f"pass@{PASS_K} (sampled at temperature {TEMPERATURE})",
             f"Is the right answer anywhere in the model's distribution across {PASS_K} tries?"),
        ]
        for i, (k, title, note) in enumerate(tables):
            if i > 0:
                print()

            def estimate(rec: dict) -> float:
                """pass@k for one case, from how many of its samples were right."""
                return pass_at_k(N_SAMPLES, samples_correct(rec), k)

            per_task = {name: per_task_scores(recs, estimate) for name, recs in records.items()}
            print_metric_table(title, note, per_task, suite.counts)

    return run_step("Step 8: pass_at_k()", show, lambda: check_pass_at_k(pass_at_k))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_CHOICES = ["1", "2", "3", "4", "5", "6", "7", "8"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small benchmark suite and score two models")
    parser.add_argument("--step", choices=[*STEP_CHOICES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = STEP_CHOICES if args.step == "all" else [args.step]

    torch.manual_seed(SEED)
    suite = load_suite()
    if args.step == "all":
        print_protocol(suite)  # a full report states its protocol before any number
    results: dict = {}  # the generated answers, shared between steps

    for step in steps:
        if step == "1":
            step_1(suite)
        elif step == "2":
            step_2()
        elif step == "3":
            step_3(suite, results)
        elif step == "4":
            step_4(suite, results)
        elif step == "5":
            step_5(suite, results)
        elif step == "6":
            step_6(suite, results)
        elif step == "7":
            step_7(suite)
        elif step == "8":
            step_8(suite, results)


if __name__ == "__main__":
    main()
