"""
Module 6 Exercise runner: Finetuning NanoGPT into an instruct model

Run with:
    uv run python module_06_finetuning/src/main.py

Loads the bundled Module 5 base checkpoint, injects LoRA adapters, finetunes on
toy instruction-response pairs, and shows the behavioral flip on one prompt:
the base model continues text and ignores the instruction; the finetuned model
answers it.

Every step is tagged on its header line, then its output follows: what your
code produced on the real model and the result of each test in tests/.
The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-10).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import copy
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable so we can grab the provided model / tokenizer / data helpers.
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
    build_generation_prompt,
    build_optimizer,
    build_targets,
    count_trainable_params,
    format_example,
    freeze_base_param,
    lora_forward_delta,
    masked_cross_entropy,
    merge_lora_weight,
    sft_train_step,
)
from data import load_dataset  # noqa: E402
from model import (  # noqa: E402
    LoRALinear,
    freeze_base_,
    generate,
    inject_lora,
    load_base_model,
    merge_lora,
)
from tokenizer import SPECIAL_TOKENS, build_vocab, decode, encode  # noqa: E402

# One test file per step lives in tests/
from tests.test_step1_format_example import check_format_example  # noqa: E402
from tests.test_step2_build_targets import check_build_targets  # noqa: E402
from tests.test_step3_masked_loss import check_masked_cross_entropy  # noqa: E402
from tests.test_step4_optimizer import check_build_optimizer  # noqa: E402
from tests.test_step5_lora_delta import check_lora_forward_delta  # noqa: E402
from tests.test_step6_freeze import check_freeze_base_param, check_freeze_on_model  # noqa: E402
from tests.test_step7_train_step import check_sft_train_step, check_sft_training  # noqa: E402
from tests.test_step8_count_params import check_count_on_model, check_count_trainable_params  # noqa: E402
from tests.test_step9_generation_prompt import check_build_generation_prompt  # noqa: E402
from tests.test_step10_merge import check_merge_lora_weight, check_merge_on_model  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to finetune on a laptop CPU in a couple minutes)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128       # context length in characters
BATCH_SIZE = 16        # examples per batch
MAX_STEPS = 1000       # total finetuning steps
EVAL_INTERVAL = 100    # report the loss every this many steps
LR = 1e-3              # small finetuning learning rate (step 4)
GRAD_CLIP = 1.0
RANK = 8               # LoRA rank r
ALPHA = 32.0           # LoRA alpha (scale = alpha / r)
DROPOUT = 0.0
SEED = 1337

NEW_VOCAB_SIZE = 65 + len(SPECIAL_TOKENS)  # 65 base chars + 4 specials = 69
SAMPLE_PROMPT = "uppercase: hello"          # the one prompt we flip before/after
SAMPLE_TOKENS = 18                          # tokens to generate per sample
SAMPLE_TEMPERATURE = 0.4                    # low temp: the toy tasks are deterministic
DEMO_PAIR = 2                               # the dataset pair Steps 1 and 2 print

_THIS_DIR = Path(__file__).resolve().parent


def _find_data_file(name: str) -> Path:
    """Walk up from this file to find data/<name>."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def load_everything() -> dict:
    """Load what every step shares: the base model, the vocabulary, and the dataset.

    Returns a dict that the steps read from and add their results to.
    """
    torch.manual_seed(SEED)

    # data/base_model.pt ships with the repo. It is the frozen Module 5 base model:
    # TinyGPT pretrained on tinyshakespeare by src/make_base_checkpoint.py.
    model, base_stoi, _ = load_base_model(_find_data_file("base_model.pt"), NEW_VOCAB_SIZE)
    stoi, itos = build_vocab(base_stoi)

    # An untouched copy of the base model, for the "before finetuning" sample
    base = copy.deepcopy(model)

    # Wrap the attention and MLP layers with LoRA adapters (provided). Every B
    # starts at zero, so the wrapped model still behaves exactly like the base.
    inject_lora(model, r=RANK, alpha=ALPHA, dropout=DROPOUT)

    return {
        "model": model,                                        # the model we finetune
        "base": base,                                          # the base model, never trained
        "dataset": load_dataset(_find_data_file("sft_pairs.jsonl")),
        "special": {tok: stoi[tok] for tok in SPECIAL_TOKENS},  # e.g. "<|user|>" -> 65
        "enc": lambda s: encode(s, stoi),                      # text -> token ids
        "dec": lambda ids: decode(ids, itos),                  # token ids -> text
    }


def _pad(seq: list[int], length: int, pad_id: int) -> list[int]:
    """Pad (or truncate) a sequence to exactly `length`."""
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_id] * (length - len(seq))


def _prompt_span(prompt: str, enc) -> int:
    """How many leading tokens belong to the prompt: user marker + text + end marker + assistant marker."""
    return 1 + len(enc(prompt)) + 1 + 1


def _build_batch(results: dict, generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one padded (x, y) batch from random prompt-response pairs.

    Targets are built from the unpadded ids so pad tokens never become targets;
    both x and y are then padded to BLOCK_SIZE with -100 padding on the targets.
    """
    pairs, special, enc = results["dataset"], results["special"], results["enc"]
    idxs = torch.randperm(len(pairs), generator=generator)[:BATCH_SIZE].tolist()
    xs, ys = [], []
    for i in idxs:
        ids = format_example(pairs[i]["prompt"], pairs[i]["response"], special, enc)
        targets = build_targets(ids, _prompt_span(pairs[i]["prompt"], enc))
        xs.append(_pad(ids, BLOCK_SIZE, special["<|pad|>"]))
        ys.append(_pad(targets, BLOCK_SIZE, -100))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def _sample(model, results: dict) -> tuple[str, str]:
    """Generate a completion for SAMPLE_PROMPT. Returns (full_text, response_only)."""
    ids = build_generation_prompt(SAMPLE_PROMPT, results["special"], results["enc"])
    idx = torch.tensor([ids], dtype=torch.long)
    gen = torch.Generator().manual_seed(SEED)
    out = generate(model, idx, SAMPLE_TOKENS, BLOCK_SIZE, temperature=SAMPLE_TEMPERATURE, generator=gen)
    full = results["dec"](out[0])
    # The response is what follows the assistant marker, up to the first <|end|>.
    after = full.split("<|assistant|>", 1)[-1]
    response = after.split("<|end|>", 1)[0]
    return full, response


# ---------------------------------------------------------------------------
# Earlier steps a later step depends on
# ---------------------------------------------------------------------------

STEP_NAMES = {
    "1": "format_example",
    "2": "build_targets",
    "3": "masked_cross_entropy",
    "4": "build_optimizer",
    "5": "lora_forward_delta",
    "6": "freeze_base_param",
    "7": "sft_train_step",
    "8": "count_trainable_params",
    "9": "build_generation_prompt",
    "10": "merge_lora_weight",
}

# One tiny call per step, used only to see whether that step is finished yet
_PROBES = {
    "1": lambda: format_example("a", "b", {"<|user|>": 0, "<|assistant|>": 1, "<|end|>": 2}, lambda s: [3]),
    "2": lambda: build_targets([0, 1, 2, 3], 2),
    "3": lambda: masked_cross_entropy(torch.zeros(1, 2, 4), torch.tensor([[-100, 1]])),
    "4": lambda: build_optimizer(nn.Linear(2, 2), LR),
    "5": lambda: lora_forward_delta(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(2, 1), 1.0, nn.Identity()),
    "6": lambda: freeze_base_param(nn.Parameter(torch.zeros(1))),
    "8": lambda: count_trainable_params(nn.Linear(1, 1)),
    "9": lambda: build_generation_prompt("a", {"<|user|>": 0, "<|assistant|>": 1, "<|end|>": 2}, lambda s: [3]),
    "10": lambda: merge_lora_weight(torch.zeros(2, 2), torch.zeros(1, 2), torch.zeros(2, 1), 1.0),
}


def _require(step: str, purpose: str) -> None:
    """Stop with a pointed message if an earlier step is still unfinished."""
    try:
        _PROBES[step]()
    except NotImplementedError:
        raise NotImplementedError(f"needs Step {step} ({STEP_NAMES[step]}) to {purpose}") from None


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
# The steps
# ---------------------------------------------------------------------------


def step_1(results: dict) -> str:
    """Format one real pair from the dataset with the chat template."""
    special, enc, dec = results["special"], results["enc"], results["dec"]

    def show():
        pair = results["dataset"][DEMO_PAIR]
        ids = format_example(pair["prompt"], pair["response"], special, enc)
        print(f"{len(results['dataset'])} instruction pairs loaded from sft_pairs.jsonl")
        print(f"Example pair: {pair['prompt']!r} -> {pair['response']!r}")
        print(f"  {len(ids)} token ids: {ids}")
        if all(isinstance(i, int) for i in ids):  # a nested list cannot be decoded
            print(f"  decoded: {dec(ids)!r}")

    return run_step("Step 1: format_example()", show,
                    lambda: check_format_example(format_example, special, enc, dec))


def step_2(results: dict) -> str:
    """Show which positions of the same pair are trained and which are masked."""

    def show():
        # Check this step first, so an unfinished build_targets() reports its own TODO
        _PROBES["2"]()
        _require("1", "format the pair it masks")

        pair = results["dataset"][DEMO_PAIR]
        ids = format_example(pair["prompt"], pair["response"], results["special"], results["enc"])
        span = _prompt_span(pair["prompt"], results["enc"])
        targets = build_targets(ids, span)
        trained = [t for t in targets if t != -100]
        print(f"Example pair: {pair['prompt']!r} -> {pair['response']!r}")
        print(f"  {len(ids)} tokens, the first {span} are the prompt (prompt_span = {span})")
        print(f"  masked to -100: {len(targets) - len(trained)} positions")
        print(f"  trained:        {len(trained)} positions, targets {results['dec'](trained)!r}")

    return run_step("Step 2: build_targets()", show, lambda: check_build_targets(build_targets))


def step_3(results: dict) -> str:
    """Score the untrained base model on one real batch with the masked loss."""

    def show():
        # This step's own function first, then the earlier steps the demo needs
        _PROBES["3"]()
        _require("1", "build a batch of formatted pairs")
        _require("2", "build the batch's masked targets")

        # Same seed as the training loop, so this is the batch that training scores at step 0
        xb, yb = _build_batch(results, torch.Generator().manual_seed(SEED))
        with torch.no_grad():
            loss = masked_cross_entropy(results["base"](xb), yb)
        n_trained = int((yb != -100).sum())
        uniform = math.log(NEW_VOCAB_SIZE)  # loss of a model that guesses every token equally
        print(f"Base model on one batch of {BATCH_SIZE} pairs ({n_trained} response tokens scored)")
        print(f"  masked loss:    {float(loss):.4f}")
        print(f"  uniform guess:  {uniform:.4f} (ln {NEW_VOCAB_SIZE})")

    return run_step("Step 3: masked_cross_entropy()", show,
                    lambda: check_masked_cross_entropy(masked_cross_entropy))


def step_4() -> str:
    """No demo: the optimizer is used for real in Step 7."""
    return run_step("Step 4: build_optimizer()", lambda: None, lambda: check_build_optimizer(build_optimizer))


def step_5(results: dict) -> str:
    """Run the LoRA-wrapped model and confirm the fresh adapters change nothing."""

    def show():
        model, base = results["model"], results["base"]
        layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
        idx = torch.tensor([results["enc"](SAMPLE_PROMPT)], dtype=torch.long)
        with torch.no_grad():
            diff = (model(idx) - base(idx)).abs().max().item()
        print(f"LoRA (r={RANK}, alpha={ALPHA:g}, scale={ALPHA / RANK:g}) wraps {len(layers)} linear layers")
        print(f"Every B starts at zero, so on {SAMPLE_PROMPT!r} the wrapped model matches the base model:")
        print(f"  max logit difference {diff:.2e}")

    return run_step("Step 5: lora_forward_delta()", show,
                    lambda: check_lora_forward_delta(lora_forward_delta))


def step_6(results: dict) -> str:
    """Freeze every base tensor of the real model (freeze_base_ calls your function on each one)."""
    model = results["model"]

    def show():
        freeze_base_(model)
        # Split the tensors into LoRA adapters (named .A and .B) and everything else
        lora = [p for name, p in model.named_parameters() if name.endswith((".A", ".B"))]
        base = [p for name, p in model.named_parameters() if not name.endswith((".A", ".B"))]
        print(f"Base tensors frozen:     {sum(not p.requires_grad for p in base)} of {len(base)} "
              "(embeddings, layer norms, and every wrapped Linear)")
        print(f"LoRA tensors trainable:  {sum(p.requires_grad for p in lora)} of {len(lora)} "
              f"(A and B in each of the {len(lora) // 2} LoRA layers)")

    return run_step("Step 6: freeze_base_param()", show,
                    lambda: check_freeze_base_param(freeze_base_param) + check_freeze_on_model(model))


def step_7(results: dict) -> str:
    """Finetune the adapters on the instruction data: the only long-running step."""
    model = results["model"]

    def show():
        # Run this step's own tests first, so an unfinished sft_train_step()
        # reports its own TODO rather than one from an earlier step
        results["step7_checks"] = check_sft_train_step(sft_train_step)

        # The training loop also runs everything from Steps 1-6
        _require("1", "build training batches")
        _require("2", "build the masked training targets")
        _require("3", "compute the training loss")
        _require("4", "build the optimizer")
        _require("5", "run the LoRA layers")
        _require("6", "freeze the base model before training")

        freeze_base_(model)                       # no-op if Step 6 already froze it
        optimizer = build_optimizer(model, LR)    # AdamW over the adapters only
        batch_gen = torch.Generator().manual_seed(SEED)
        losses = []
        print(f"{'step':>6}  {'loss':>8}")
        for step in range(MAX_STEPS + 1):
            if step % EVAL_INTERVAL == 0:
                # Report the loss on a fresh batch, without training on it
                with torch.no_grad():
                    xb, yb = _build_batch(results, batch_gen)
                    loss = float(masked_cross_entropy(model(xb), yb))
                losses.append(loss)
                print(f"{step:>6}  {loss:>8.4f}")
            if step == MAX_STEPS:
                break
            xb, yb = _build_batch(results, batch_gen)
            sft_train_step(model, optimizer, xb, yb, GRAD_CLIP)
        results["losses"] = losses

    return run_step("Step 7: sft_train_step()", show,
                    lambda: results["step7_checks"] + check_sft_training(results))


def step_8(results: dict) -> str:
    """Count what LoRA actually trains, against the size of the whole model."""
    model = results["model"]

    def show():
        # This step's own function first, then the earlier step the demo needs
        _PROBES["8"]()
        _require("6", "freeze the base before counting what still trains")
        freeze_base_(model)  # no-op if Step 6 already froze it
        trainable = count_trainable_params(model)
        total = model.num_params()
        print(f"Trainable (LoRA adapters):  {trainable:,}")
        print(f"Total (base + adapters):    {total:,}")
        print(f"Fraction trainable:         {trainable / total:.2%}")

    return run_step("Step 8: count_trainable_params()", show,
                    lambda: check_count_trainable_params(count_trainable_params)
                    + check_count_on_model(count_trainable_params, model))


def step_9(results: dict) -> str:
    """The same instruction, before and after finetuning: the behavioral flip."""
    special, enc, dec = results["special"], results["enc"], results["dec"]

    def show():
        # The base model has no adapters, so it only needs this step's prompt builder
        full, response = _sample(results["base"], results)
        print(f"prompt: {SAMPLE_PROMPT!r}")
        print("Base model (before finetuning):")
        print(f"  full:     {full!r}")
        print(f"  response: {response!r}   <- ignores the instruction")

        print("Finetuned model (after Step 7):")
        if "losses" not in results:
            print("  not finetuned in this run (Step 7 has not trained the adapters)")
            return
        full, response = _sample(results["model"], results)
        print(f"  full:     {full!r}")
        print(f"  response: {response!r}   <- answers the instruction")

    return run_step("Step 9: build_generation_prompt()", show,
                    lambda: check_build_generation_prompt(build_generation_prompt, special, enc, dec))


def step_10(results: dict) -> str:
    """Fold every adapter back into its weight matrix and check that nothing changed."""
    model = results["model"]

    def show():
        # This step's own function first, then the earlier step the demo needs
        _PROBES["10"]()
        _require("5", "run the adapter model the merge is compared against")

        # Merge a deep copy so the adapter model is left intact for the comparison
        merged = merge_lora(copy.deepcopy(model))
        n_layers = sum(isinstance(m, LoRALinear) for m in model.modules())
        idx = torch.tensor([results["enc"](SAMPLE_PROMPT)], dtype=torch.long)
        with torch.no_grad():
            diff = (model(idx) - merged(idx)).abs().max().item()
        results["merge_diff"] = diff
        print(f"Merged {n_layers} LoRA layers into plain nn.Linear weights: "
              f"{model.num_params():,} -> {merged.num_params():,} parameters")
        print(f"  Max logit difference (adapter vs merged): {diff:.2e}")
        if "losses" not in results:
            print("  (Step 7 has not trained the adapters in this run, so every B is still zero)")

    return run_step("Step 10: merge_lora_weight()", show,
                    lambda: check_merge_lora_weight(merge_lora_weight) + check_merge_on_model(results))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Finetuning NanoGPT into an instruct model")
    parser.add_argument("--step", choices=[*STEP_NAMES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = list(STEP_NAMES) if args.step == "all" else [args.step]

    results = load_everything()  # the model, vocabulary, and data, shared by every step

    for step in steps:
        if step == "1":
            step_1(results)
        elif step == "2":
            step_2(results)
        elif step == "3":
            step_3(results)
        elif step == "4":
            step_4()
        elif step == "5":
            step_5(results)
        elif step == "6":
            step_6(results)
        elif step == "7":
            step_7(results)
        elif step == "8":
            step_8(results)
        elif step == "9":
            step_9(results)
        elif step == "10":
            step_10(results)


if __name__ == "__main__":
    main()
