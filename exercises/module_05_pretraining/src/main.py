"""
Module 5 Exercise runner: Pretraining NanoGPT

Run with:
    uv run python module_05_pretraining/src/main.py

Trains a tiny decoder-only language model from scratch on a bundled text file.
The goal is not a useful model; it is to make the pretraining loop visible:
loss going down, validation loss tracking it, perplexity and bits per token
falling, and samples improving from random characters to text-like output.

Every step is tagged on its header line, then its output follows: what your
code produced and the result of each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1, 2, 3, 4, 5, 7, 8, or 10). Steps 1 and 2
always run first, because the other steps need the token streams they build.
Add --overfit to run the single-batch sanity check instead of full training.
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable so we can grab the provided model + visualization helpers.
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
    compute_loss,
    encode,
    estimate_loss,
    generate,
    get_batch,
    loss_to_perplexity_and_bits,
    train_step,
    train_val_split,
)
from model import GPTConfig, TinyGPT  # noqa: E402
from src.schedules import lr_at_step  # noqa: E402

# One test file per step lives in tests/
from tests.test_step1_encode import check_encode  # noqa: E402
from tests.test_step2_split import check_train_val_split  # noqa: E402
from tests.test_step3_get_batch import check_get_batch  # noqa: E402
from tests.test_step4_loss import check_compute_loss  # noqa: E402
from tests.test_step5_train_step import check_overfit, check_train_step  # noqa: E402
from tests.test_step7_estimate_loss import check_estimate_loss, check_pretraining  # noqa: E402
from tests.test_step8_perplexity import check_loss_to_perplexity_and_bits  # noqa: E402
from tests.test_step10_generate import check_generate  # noqa: E402
from visualization import plot_loss_curve  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to train on a laptop CPU in a few minutes)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128       # context length in characters
BATCH_SIZE = 32        # examples per batch
N_LAYER = 4            # transformer blocks
N_HEAD = 4             # attention heads per block
N_EMBD = 128           # model width
DROPOUT = 0.1

MAX_STEPS = 2000       # total training steps
WARMUP_STEPS = 100     # linear LR warmup
MAX_LR = 3e-3          # peak learning rate
MIN_LR = 3e-4          # final learning rate
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250    # estimate train/val loss every this many steps
EVAL_BATCHES = 20      # batches averaged per loss estimate
VAL_FRACTION = 0.1
SEED = 1337

SAMPLE_TOKENS = 300    # characters to generate for the before/after samples
SAMPLE_SEED_TEXT = "\n"  # what to prime generation with

OVERFIT_BATCH_SIZE = 8  # small fixed batch for the --overfit sanity check
OVERFIT_LR = 3e-3       # learning rate for the overfit sanity check
OVERFIT_STEPS = 300     # optimizer steps on that one batch

_THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _THIS_DIR.parent / "output"


def _find_data_file() -> Path:
    """Walk up from this file to find data/tinyshakespeare.txt."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / "tinyshakespeare.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate data/tinyshakespeare.txt")


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
# Training helpers (shared by the steps below)
# ---------------------------------------------------------------------------

def _seed_torch() -> None:
    """Seed PyTorch's random numbers so every run builds the same model.

    After seeding, this draws the same random numbers that an earlier version
    of this runner drew for a quick self-test on a scratch model. The draws
    are thrown away. They only keep the starting weights, and so every number
    and sample shown in the lecture slides, identical to the captured run.
    """
    torch.manual_seed(SEED)
    scratch = TinyGPT(GPTConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=16))
    for shape in [(64,), (2, 8), (2, 8)]:
        torch.randint(0, 7, shape)
    torch.randn(2, 8, 7)
    torch.randint(56, (2,))                              # one batch of start indices
    scratch(torch.zeros(2, 8, dtype=torch.long))         # one forward pass with dropout on
    torch.randint(56, (2,))                              # another batch of start indices
    for _ in range(2):
        torch.multinomial(torch.full((1, 7), 1 / 7), 1)  # two sampled tokens


def _finished(fn, *args) -> bool:
    """True unless fn raises NotImplementedError on these small inputs."""
    try:
        fn(*args)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - the student wrote *something*; its own step reports the crash
        return True
    return True


def _needs(done: bool, message: str) -> None:
    """Stop a step's demo with a pointed message when an earlier step is missing."""
    if not done:
        raise NotImplementedError(message)


def _placeholder_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stand-in for compute_loss(): a zero that autograd can backpropagate (not the answer)."""
    return logits.sum() * 0.0


def _placeholder_batch(data, block_size, batch_size, generator=None):
    """Stand-in for get_batch(): the first block, repeated (not the answer)."""
    x = data[:block_size].repeat(batch_size, 1)
    return x, x


def _try_own(fn, *args, **placeholders) -> None:
    """Call a step's own function once on tiny inputs, so its own TODO shows first.

    `placeholders` replace the earlier-step functions it calls (such as
    compute_loss) for this one call, so an unfinished earlier step cannot hide
    this step's TODO. A crash is ignored here; the real demo reports it.
    """
    namespace = inspect.unwrap(fn).__globals__  # the exercise module's variables
    saved = {name: namespace[name] for name in placeholders}
    namespace.update(placeholders)
    try:
        fn(*args)
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001 - reported by the real demo instead
        pass
    finally:
        namespace.update(saved)  # put the real functions back


def _tiny_model() -> nn.Embedding:
    """A 3-token bigram model of zeros: just enough to call a function once."""
    return nn.Embedding.from_pretrained(torch.zeros(3, 3), freeze=False)


def _tiny_ids() -> torch.Tensor:
    """A short stream of token IDs (0, 1, 2, 0, 1, 2, ...) for the tiny model."""
    return torch.arange(12) % 3


def _get_batch_done() -> bool:
    """Is Step 3 (get_batch) filled in? Tried on a tiny stream with its own generator."""
    return _finished(get_batch, _tiny_ids(), 4, 1, torch.Generator().manual_seed(0))


def _compute_loss_done() -> bool:
    """Is Step 4 (compute_loss) filled in? Tried on one position of 3 logits."""
    return _finished(compute_loss, torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long))


def _train_step_done() -> bool:
    """Is Step 5 (train_step) filled in? Tried on the tiny model, so the real one is untouched."""
    model = _tiny_model()
    ids = _tiny_ids()[:4].view(1, 4)
    return _finished(train_step, model, torch.optim.SGD(model.parameters(), lr=0.1), ids, ids)


def _decode(itos: dict[int, str], ids: torch.Tensor) -> str:
    """Turn a 1-D tensor of token IDs back into text."""
    return "".join(itos[int(i)] for i in ids)


def _sample(model, results: dict, generator: torch.Generator) -> str:
    """Generate SAMPLE_TOKENS characters after the seed text (uses your generate())."""
    seed_ids = torch.tensor([[results["stoi"][c] for c in SAMPLE_SEED_TEXT]], dtype=torch.long)
    out = generate(model, seed_ids, SAMPLE_TOKENS, BLOCK_SIZE, temperature=1.0, generator=generator)
    # Drop the seed text so only the model's own characters are shown
    return _decode(results["itos"], out[0])[len(SAMPLE_SEED_TEXT):]


def _print_text(text: str) -> None:
    """Print generated text indented under its label, one line at a time."""
    for line in text.split("\n"):
        print(f"    {line}")


def _progress(message: str) -> None:
    """Show a live progress line on a real terminal (the step's output is held until it ends)."""
    if sys.__stderr__.isatty():
        sys.__stderr__.write(f"\r  {message}\033[K")
        sys.__stderr__.flush()


def _progress_done() -> None:
    """Erase the live progress line."""
    if sys.__stderr__.isatty():
        sys.__stderr__.write("\r\033[K")
        sys.__stderr__.flush()


def pretrain(results: dict) -> None:
    """The full pretraining run: Steps 3, 4, 5, and 7 together, with the provided schedule.

    Stores the checkpoint history in `results` for Steps 8 and 10.
    """
    model = results["model"]
    train_data, val_data = results["train_data"], results["val_data"]

    # The tests above drew random numbers too. Rewind to the state right after
    # the model was built, so training is the same on every run.
    torch.set_rng_state(results["rng_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    batch_gen = torch.Generator().manual_seed(SEED)
    eval_gen = torch.Generator().manual_seed(SEED + 1)
    ckpt_steps: list[int] = []
    train_hist: list[float] = []
    val_hist: list[float] = []

    print(f"Pretraining for {MAX_STEPS:,} steps (warmup + cosine learning rate from src/schedules.py):")
    print(f"{'step':>6}  {'lr':>9}  {'train':>8}  {'val':>8}")
    for step in range(MAX_STEPS + 1):
        # Set this step's learning rate from the schedule (provided).
        lr = lr_at_step(step, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodically estimate train/val loss (Step 7) and record a checkpoint.
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            tr = estimate_loss(model, train_data, BLOCK_SIZE, BATCH_SIZE, EVAL_BATCHES, eval_gen)
            va = estimate_loss(model, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_BATCHES, eval_gen)
            ckpt_steps.append(step)
            train_hist.append(tr)
            val_hist.append(va)
            print(f"{step:>6}  {lr:>9.2e}  {tr:>8.4f}  {va:>8.4f}")

        if step == MAX_STEPS:
            break
        if step % 50 == 0:
            _progress(f"training: step {step:,} of {MAX_STEPS:,}, validation loss {val_hist[-1]:.4f}")

        # One optimizer step on a fresh batch (Steps 3 + 4 + 5).
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, batch_gen)
        train_step(model, optimizer, x, y, GRAD_CLIP)
    _progress_done()

    results["ckpt_steps"], results["train_hist"], results["val_hist"] = ckpt_steps, train_hist, val_hist

    # Plot the loss curve from the recorded checkpoints (provided).
    plot_loss_curve(ckpt_steps, train_hist, val_hist, str(OUTPUT_DIR / "loss_curve.png"))


def overfit(results: dict) -> None:
    """Single-batch sanity check: the loss on one fixed batch should fall toward 0."""
    model = results["model"]
    torch.set_rng_state(results["rng_state"])  # same starting point as the full run

    gen = torch.Generator().manual_seed(SEED)
    # A small fixed batch the model can memorize, so the loss should crater to ~0.
    x, y = get_batch(results["train_data"], BLOCK_SIZE, OVERFIT_BATCH_SIZE, gen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    losses: list[float] = []
    print(f"Training repeatedly on ONE batch of shape {tuple(x.shape)} for {OVERFIT_STEPS} steps:")
    print(f"{'step':>6}  {'loss':>8}")
    for step in range(OVERFIT_STEPS + 1):
        loss = train_step(model, optimizer, x, y, GRAD_CLIP)
        losses.append(loss)
        if step % 50 == 0:
            print(f"{step:>6}  {loss:>8.4f}")
            _progress(f"overfitting: step {step} of {OVERFIT_STEPS}, loss {loss:.4f}")
    _progress_done()
    results["overfit_losses"] = losses


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def step_1(results: dict) -> str:
    """Encode the whole corpus. Later steps read it from results["data"]."""

    def show():
        data = encode(results["text"], results["stoi"])
        results["data"] = data
        print(f"  Encoded {len(data):,} tokens. First 20 IDs: {data[:20].tolist()}")

    return run_step("Step 1: encode()", show,
                    lambda: check_encode(encode, results["text"], results["stoi"]))


def step_2(results: dict) -> str:
    """Split the token stream. Later steps read results["train_data"] and ["val_data"]."""

    def show():
        train_val_split(torch.arange(10))  # your own TODO first
        _needs("data" in results, "needs Step 1 (encode) to turn the corpus into token IDs")
        train_data, val_data = train_val_split(results["data"], VAL_FRACTION)
        results["train_data"], results["val_data"] = train_data, val_data
        print(f"  Train tokens: {len(train_data):,}   Validation tokens: {len(val_data):,}")

    return run_step("Step 2: train_val_split()", show,
                    lambda: check_train_val_split(train_val_split))


def step_3(results: dict) -> str:
    """Draw one real batch and show that y is x shifted by one character."""

    def show():
        get_batch(_tiny_ids(), 4, 1, torch.Generator().manual_seed(0))  # your own TODO first
        _needs("train_data" in results, "needs Step 2 (train_val_split) for the training stream")
        gen = torch.Generator().manual_seed(SEED)
        x, y = get_batch(results["train_data"], BLOCK_SIZE, BATCH_SIZE, gen)
        itos = results["itos"]
        print(f"  One batch: x has shape {tuple(x.shape)}, y has shape {tuple(y.shape)}")
        print(f"  x[0][:24] = {_decode(itos, x[0][:24])!r}")
        print(f"  y[0][:24] = {_decode(itos, y[0][:24])!r}")

    return run_step("Step 3: get_batch()", show, lambda: check_get_batch(get_batch))


def step_4(results: dict) -> str:
    """Score the untrained model on real text: close to a uniform guess."""

    def show():
        compute_loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long))  # your own TODO first
        _needs("train_data" in results, "needs Step 2 (train_val_split) for the training stream")
        # The first 8 x 128 characters as inputs, and the same text shifted by one as targets
        n = OVERFIT_BATCH_SIZE * BLOCK_SIZE
        x = results["train_data"][:n].view(OVERFIT_BATCH_SIZE, BLOCK_SIZE)
        y = results["train_data"][1 : n + 1].view(OVERFIT_BATCH_SIZE, BLOCK_SIZE)
        model = results["untrained"]
        model.eval()  # turn dropout off while measuring
        with torch.no_grad():
            loss = compute_loss(model(x), y)
        model.train()
        vocab_size = len(results["stoi"])
        print(f"  Untrained model on {OVERFIT_BATCH_SIZE} x {BLOCK_SIZE} characters: loss {float(loss):.4f} nats")
        print(f"  A uniform guess over {vocab_size} characters costs ln {vocab_size} = {math.log(vocab_size):.4f} nats")

    return run_step("Step 4: compute_loss()", show, lambda: check_compute_loss(compute_loss))


def step_5(results: dict) -> str:
    """A few optimizer steps on one real batch, on a throwaway copy of the model."""

    def show():
        # Your own TODO first (with a placeholder loss), then the Step 4 it relies on
        tiny, ids = _tiny_model(), _tiny_ids()[:4].view(1, 4)
        _try_own(train_step, tiny, torch.optim.SGD(tiny.parameters(), lr=0.1), ids, ids,
                 compute_loss=_placeholder_loss)
        _needs(_compute_loss_done(), "needs Step 4 (compute_loss) to compute the loss it backpropagates")
        _needs("train_data" in results, "needs Step 2 (train_val_split) for the training stream")
        n = OVERFIT_BATCH_SIZE * BLOCK_SIZE
        x = results["train_data"][:n].view(OVERFIT_BATCH_SIZE, BLOCK_SIZE)
        y = results["train_data"][1 : n + 1].view(OVERFIT_BATCH_SIZE, BLOCK_SIZE)
        model = copy.deepcopy(results["untrained"])  # a copy, so the real model stays untrained
        optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR)
        losses = [train_step(model, optimizer, x, y, GRAD_CLIP) for _ in range(5)]
        print(f"  Five train_step() calls on one batch of {OVERFIT_BATCH_SIZE} x {BLOCK_SIZE} characters:")
        print("  loss " + " -> ".join(f"{loss:.4f}" for loss in losses))

    return run_step("Step 5: train_step()", show, lambda: check_train_step(train_step))


def step_overfit(results: dict) -> str:
    """The --overfit sanity check, run in place of Steps 7-10."""

    def show():
        _needs(_get_batch_done(), "needs Step 3 (get_batch) to draw the batch it memorizes")
        _needs(_compute_loss_done(), "needs Step 4 (compute_loss) to measure the loss")
        _needs(_train_step_done(), "needs Step 5 (train_step) to take the optimizer steps")
        _needs("train_data" in results, "needs Step 2 (train_val_split) for the training stream")
        overfit(results)

    return run_step("Sanity check: overfit one batch (Steps 3-5 together)", show,
                    lambda: check_overfit(results))


def step_7(results: dict) -> str:
    """Test estimate_loss(), then run the full pretraining loop (Steps 3-7 together)."""

    def show():
        # Your own TODO first (with placeholders), then the Steps 3 and 4 it relies on
        _try_own(estimate_loss, _tiny_model(), _tiny_ids(), 4, 1, 1, torch.Generator().manual_seed(0),
                 get_batch=_placeholder_batch, compute_loss=_placeholder_loss)
        _needs(_get_batch_done(), "needs Step 3 (get_batch) to draw the batches it averages")
        _needs(_compute_loss_done(), "needs Step 4 (compute_loss) to score each batch")
        _needs(_train_step_done(), "needs Step 5 (train_step) to run the training loop")
        _needs("val_data" in results, "needs Step 2 (train_val_split) for the training and validation streams")
        pretrain(results)

    return run_step("Step 7: estimate_loss()", show,
                    lambda: check_estimate_loss(estimate_loss) + check_pretraining(results))


def step_8(results: dict) -> str:
    """Read the validation loss as perplexity and bits per token."""

    def show():
        vocab_size = len(results["stoi"])
        rows = [(f"uniform over {vocab_size} chars", math.log(vocab_size))]
        if "val_hist" in results:
            rows.append(("before training", results["val_hist"][0]))
            rows.append(("after training", results["val_hist"][-1]))
        # Convert every row before printing, so an unfinished Step 8 prints nothing
        readouts = [(label, loss, *loss_to_perplexity_and_bits(loss)) for label, loss in rows]

        print("  Validation loss, read three ways:")
        print(f"  {'':<22}{'nats':>8}{'perplexity':>13}{'bits/token':>13}")
        for label, loss, ppl, bits in readouts:
            print(f"  {label:<22}{loss:>8.4f}{ppl:>13.2f}{bits:>13.4f}")
        if "val_hist" not in results:
            print("  (the before and after rows need the training run from Step 7)")

    return run_step("Step 8: loss_to_perplexity_and_bits()", show,
                    lambda: check_loss_to_perplexity_and_bits(loss_to_perplexity_and_bits))


def step_10(results: dict) -> str:
    """Sample text from the model before and after training."""

    def show():
        # The same seeded generator for both samples, as in the captured run
        gen = torch.Generator().manual_seed(SEED)
        before = _sample(results["untrained"], results, gen)
        print("  Sample before training (random weights):")
        _print_text(before)
        if "val_hist" in results:
            after = _sample(results["model"], results, gen)
            print(f"  Sample after training ({MAX_STEPS:,} steps):")
            _print_text(after)
        else:
            print("  (the after-training sample needs the training run from Step 7)")

    return run_step("Step 10: generate()", show, lambda: check_generate(generate))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {
    "1": step_1,
    "2": step_2,
    "3": step_3,
    "4": step_4,
    "5": step_5,
    "7": step_7,
    "8": step_8,
    "10": step_10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a tiny GPT.")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all). Steps 1 and 2 always run first.")
    parser.add_argument("--overfit", action="store_true",
                        help="run the single-batch overfit sanity check instead of full training")
    args = parser.parse_args()

    # Steps 1 and 2 always run: every later step needs the token streams they build
    steps = list(STEPS) if args.step == "all" else ["1", "2"]
    if args.step not in ("all", "1", "2"):
        steps.append(args.step)
    if args.overfit:
        # The sanity check replaces the full training run (Step 7) and what depends on it
        steps = [s for s in steps if s in ("1", "2", "3", "4", "5")] + ["overfit"]

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data: load the corpus and build a character-level vocabulary.
    # (No student code needed for this part.)
    # ------------------------------------------------------------------
    data_file = _find_data_file()
    text = data_file.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"Corpus:      {data_file.name}  ({len(text):,} characters)")
    print(f"Vocabulary:  {len(chars)} unique characters")

    # ------------------------------------------------------------------
    # Build the model (provided) and report its size.
    # ------------------------------------------------------------------
    _seed_torch()
    cfg = GPTConfig(vocab_size=len(chars), block_size=BLOCK_SIZE,
                    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT)
    model = TinyGPT(cfg)
    print(f"TinyGPT:     {N_LAYER} layers, {N_HEAD} heads, width {N_EMBD}, context {BLOCK_SIZE}")
    print(f"Parameters:  {model.num_params():,}")
    print()

    # Everything the steps share: the corpus, the vocabulary, the model to
    # train, an untouched copy of it, and the random state training starts from
    results: dict = {
        "text": text,
        "stoi": stoi,
        "itos": itos,
        "model": model,
        "untrained": copy.deepcopy(model),
        "rng_state": torch.get_rng_state(),
    }

    for step in steps:
        if step == "overfit":
            step_overfit(results)
        else:
            STEPS[step](results)


if __name__ == "__main__":
    main()
