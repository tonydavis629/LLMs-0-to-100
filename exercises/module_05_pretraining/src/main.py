"""
Module 5 Exercise runner: Pretraining NanoGPT

Run with:
    uv run python module_05_pretraining/src/main.py

Trains a tiny decoder-only language model from scratch on a bundled text file.
The goal is not a useful model; it is to make the pretraining loop visible:
loss going down, validation loss tracking it, perplexity and bits per token
falling, and samples improving from random characters to text-like output.

Read top to bottom, the steps follow the data through pretraining:

    1. text -> token IDs
    2. token IDs -> a training stream and a validation stream
    3. a random batch of inputs x and targets y (y is x shifted by one)
    4. the loss: how surprised the model is by the real next characters
    5. one training step: backpropagate the loss, nudge the weights
    7. estimating train and validation loss, then the full pretraining run
    8. reading the loss as perplexity and bits per token
   10. generating text before and after training

Add --step N to run one step (1, 2, 3, 4, 5, 7, 8, or 10). Steps 1 and 2
always run first, because the other steps need the token streams they build.
Add --overfit to run the single-batch sanity check instead of full training.
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    compute_loss,
    encode,
    estimate_loss,
    generate,
    get_batch,
    loss_to_perplexity_and_bits,
    train_step,
    train_val_split,
)
from src.corpus import decode, find_data_file, print_indented
from src.model import GPTConfig, TinyGPT
from src.prerequisites import require, try_own_blank
from src.reporting import progress, progress_done, run_step
from src.schedules import lr_at_step
from src.seeding import seed_like_captured_run
from src.visualization import plot_loss_curve

# One test file per step lives in tests/
from tests.test_step1_encode import check_encode
from tests.test_step2_split import check_train_val_split
from tests.test_step3_get_batch import check_get_batch
from tests.test_step4_loss import check_compute_loss
from tests.test_step5_train_step import check_overfit, check_train_step
from tests.test_step7_estimate_loss import check_estimate_loss, check_pretraining
from tests.test_step8_perplexity import check_loss_to_perplexity_and_bits
from tests.test_step10_generate import check_generate

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ---------------------------------------------------------------------------
# Hyperparameters (small enough to train on a laptop CPU in under a minute)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 32        # context length in characters
BATCH_SIZE = 32        # examples per batch
N_LAYER = 2            # transformer blocks
N_HEAD = 4             # attention heads per block
N_EMBD = 64            # model width
DROPOUT = 0.0          # no dropout: this short run sees each character less than once

MAX_STEPS = 800        # total training steps
WARMUP_STEPS = 40      # linear LR warmup
MAX_LR = 6e-3          # peak learning rate
MIN_LR = 6e-4          # final learning rate
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100    # estimate train/val loss every this many steps
EVAL_BATCHES = 20      # batches averaged per loss estimate
VAL_FRACTION = 0.1
SEED = 1337

SAMPLE_TOKENS = 300    # characters to generate for the before/after samples
SAMPLE_SEED_TEXT = "\n"  # what to prime generation with

OVERFIT_BATCH_SIZE = 8  # small fixed batch for the --overfit sanity check
OVERFIT_LR = 3e-3       # learning rate for the overfit sanity check
OVERFIT_STEPS = 100     # optimizer steps on that one batch

# Everything the steps hand to each other: the corpus and vocabulary, the
# model, and what each step builds (token IDs, splits, loss history).
# main() fills in the first few entries; the steps add the rest.
shared: dict = {}


def first_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first batch_size x BLOCK_SIZE training characters, and the same text shifted by one."""
    n = batch_size * BLOCK_SIZE
    x = shared["train_data"][:n].view(batch_size, BLOCK_SIZE)
    y = shared["train_data"][1 : n + 1].view(batch_size, BLOCK_SIZE)
    return x, y


def sample(model, generator: torch.Generator) -> str:
    """Generate SAMPLE_TOKENS characters after the seed text (uses your generate())."""
    seed_ids = torch.tensor([[shared["stoi"][c] for c in SAMPLE_SEED_TEXT]], dtype=torch.long)
    out = generate(model, seed_ids, SAMPLE_TOKENS, BLOCK_SIZE, temperature=1.0, generator=generator)
    # Drop the seed text so only the model's own characters are shown
    return decode(shared["itos"], out[0])[len(SAMPLE_SEED_TEXT):]


# ---------------------------------------------------------------------------
# The pretraining loop: Steps 3, 4, 5 and 7 working together
# ---------------------------------------------------------------------------


def pretrain() -> list[float]:
    """The full pretraining run, with the provided learning-rate schedule.

    Returns the validation loss at each checkpoint, first to last.
    """
    model = shared["model"]
    train_data, val_data = shared["train_data"], shared["val_data"]

    # The tests above drew random numbers too. Rewind to the state right after
    # the model was built, so training is the same on every run.
    torch.set_rng_state(shared["rng_state"])

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
            progress(f"training: step {step:,} of {MAX_STEPS:,}, validation loss {val_hist[-1]:.4f}")

        # One optimizer step on a fresh batch (Steps 3 + 4 + 5).
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, batch_gen)
        train_step(model, optimizer, x, y, GRAD_CLIP)
    progress_done()

    # Plot the loss curve from the recorded checkpoints (provided).
    plot_loss_curve(ckpt_steps, train_hist, val_hist, str(OUTPUT_DIR / "loss_curve.png"))
    return val_hist


def overfit() -> list[float]:
    """Single-batch sanity check: the loss on one fixed batch should fall toward 0.

    Returns the loss after every step.
    """
    model = shared["model"]
    torch.set_rng_state(shared["rng_state"])  # same starting point as the full run

    gen = torch.Generator().manual_seed(SEED)
    # A small fixed batch the model can memorize, so the loss should crater to ~0.
    x, y = get_batch(shared["train_data"], BLOCK_SIZE, OVERFIT_BATCH_SIZE, gen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    losses: list[float] = []
    print(f"Training repeatedly on ONE batch of shape {tuple(x.shape)} for {OVERFIT_STEPS} steps:")
    print(f"{'step':>6}  {'loss':>8}")
    for step in range(OVERFIT_STEPS + 1):
        loss = train_step(model, optimizer, x, y, GRAD_CLIP)
        losses.append(loss)
        if step % 20 == 0:
            print(f"{step:>6}  {loss:>8.4f}")
            progress(f"overfitting: step {step} of {OVERFIT_STEPS}, loss {loss:.4f}")
    progress_done()
    return losses


# ---------------------------------------------------------------------------
# Step 1: text -> token IDs
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        data = encode(shared["text"], shared["stoi"])
        shared["data"] = data  # later steps read the token IDs from here
        print(f"  Encoded {len(data):,} tokens. First 20 IDs: {data[:20].tolist()}")

    return run_step("Step 1: encode()", show,
                    lambda: check_encode(encode, shared["text"], shared["stoi"]))


# ---------------------------------------------------------------------------
# Step 2: a training stream and a validation stream
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        try_own_blank("2")
        require("1", "to turn the corpus into token IDs", done="data" in shared)
        train_data, val_data = train_val_split(shared["data"], VAL_FRACTION)
        shared["train_data"], shared["val_data"] = train_data, val_data
        print(f"  Train tokens: {len(train_data):,}   Validation tokens: {len(val_data):,}")

    return run_step("Step 2: train_val_split()", show,
                    lambda: check_train_val_split(train_val_split))


# ---------------------------------------------------------------------------
# Step 3: one random batch, where y is x shifted by one character
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        try_own_blank("3")
        require("2", "for the training stream", done="train_data" in shared)
        gen = torch.Generator().manual_seed(SEED)
        x, y = get_batch(shared["train_data"], BLOCK_SIZE, BATCH_SIZE, gen)
        print(f"  One batch: x has shape {tuple(x.shape)}, y has shape {tuple(y.shape)}")
        print(f"  x[0][:24] = {decode(shared['itos'], x[0][:24])!r}")
        print(f"  y[0][:24] = {decode(shared['itos'], y[0][:24])!r}")

    return run_step("Step 3: get_batch()", show, lambda: check_get_batch(get_batch))


# ---------------------------------------------------------------------------
# Step 4: the loss of the untrained model, close to a uniform guess
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        try_own_blank("4")
        require("2", "for the training stream", done="train_data" in shared)
        x, y = first_batch(OVERFIT_BATCH_SIZE)
        model = shared["untrained"]
        model.eval()  # turn dropout off while measuring
        with torch.no_grad():
            loss = compute_loss(model(x), y)
        model.train()
        vocab_size = len(shared["stoi"])
        print(f"  Untrained model on {OVERFIT_BATCH_SIZE} x {BLOCK_SIZE} characters: loss {float(loss):.4f} nats")
        print(f"  A uniform guess over {vocab_size} characters costs ln {vocab_size} = {math.log(vocab_size):.4f} nats")

    return run_step("Step 4: compute_loss()", show, lambda: check_compute_loss(compute_loss))


# ---------------------------------------------------------------------------
# Step 5: a few training steps on one batch
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        try_own_blank("5")
        require("4", "to compute the loss it backpropagates")
        require("2", "for the training stream", done="train_data" in shared)
        x, y = first_batch(OVERFIT_BATCH_SIZE)
        model = copy.deepcopy(shared["untrained"])  # a copy, so the real model stays untrained
        optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR)
        losses = [train_step(model, optimizer, x, y, GRAD_CLIP) for _ in range(5)]
        print(f"  Five train_step() calls on one batch of {OVERFIT_BATCH_SIZE} x {BLOCK_SIZE} characters:")
        print("  loss " + " -> ".join(f"{loss:.4f}" for loss in losses))

    return run_step("Step 5: train_step()", show, lambda: check_train_step(train_step))


# ---------------------------------------------------------------------------
# --overfit: the single-batch sanity check, run in place of Steps 7-10
# ---------------------------------------------------------------------------


def step_overfit() -> str:
    def show():
        require("3", "to draw the batch it memorizes")
        require("4", "to measure the loss")
        require("5", "to take the optimizer steps")
        require("2", "for the training stream", done="train_data" in shared)
        shared["overfit_losses"] = overfit()

    return run_step("Sanity check: overfit one batch (Steps 3-5 together)", show,
                    lambda: check_overfit(shared["overfit_losses"]))


# ---------------------------------------------------------------------------
# Step 7: estimating the loss, then the full pretraining run
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        try_own_blank("7")
        require("3", "to draw the batches it averages")
        require("4", "to score each batch")
        require("5", "to run the training loop")
        require("2", "for the training and validation streams", done="val_data" in shared)
        shared["val_hist"] = pretrain()  # Steps 8 and 10 read the loss history from here

    return run_step("Step 7: estimate_loss()", show,
                    lambda: check_estimate_loss(estimate_loss) + check_pretraining(shared["val_hist"]))


# ---------------------------------------------------------------------------
# Step 8: the validation loss as perplexity and bits per token
# ---------------------------------------------------------------------------


def step_8() -> str:
    def show():
        vocab_size = len(shared["stoi"])
        rows = [(f"uniform over {vocab_size} chars", math.log(vocab_size))]
        if "val_hist" in shared:
            rows.append(("before training", shared["val_hist"][0]))
            rows.append(("after training", shared["val_hist"][-1]))
        # Convert every row before printing, so an unfinished Step 8 prints nothing
        readouts = [(label, loss, *loss_to_perplexity_and_bits(loss)) for label, loss in rows]

        print("  Validation loss, read three ways:")
        print(f"  {'':<22}{'nats':>8}{'perplexity':>13}{'bits/token':>13}")
        for label, loss, ppl, bits in readouts:
            print(f"  {label:<22}{loss:>8.4f}{ppl:>13.2f}{bits:>13.4f}")
        if "val_hist" not in shared:
            print("  (the before and after rows need the training run from Step 7)")

    return run_step("Step 8: loss_to_perplexity_and_bits()", show,
                    lambda: check_loss_to_perplexity_and_bits(loss_to_perplexity_and_bits))


# ---------------------------------------------------------------------------
# Step 10: text from the model before and after training
# ---------------------------------------------------------------------------


def step_10() -> str:
    def show():
        # The same seeded generator for both samples, as in the captured run
        gen = torch.Generator().manual_seed(SEED)
        print("  Sample before training (random weights):")
        print_indented(sample(shared["untrained"], gen))
        if "val_hist" in shared:
            print(f"  Sample after training ({MAX_STEPS:,} steps):")
            print_indented(sample(shared["model"], gen))
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

    # Load the corpus and build a character-level vocabulary: one ID per character
    data_file = find_data_file()
    text = data_file.read_text(encoding="utf-8")
    chars = sorted(set(text))
    shared["text"] = text
    shared["stoi"] = {c: i for i, c in enumerate(chars)}  # character -> ID
    shared["itos"] = {i: c for i, c in enumerate(chars)}  # ID -> character
    print(f"Corpus:      {data_file.name}  ({len(text):,} characters)")
    print(f"Vocabulary:  {len(chars)} unique characters")

    # Build the model (provided in src/model.py) and report its size
    seed_like_captured_run(SEED)
    cfg = GPTConfig(vocab_size=len(chars), block_size=BLOCK_SIZE,
                    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT)
    model = TinyGPT(cfg)
    print(f"TinyGPT:     {N_LAYER} layers, {N_HEAD} heads, width {N_EMBD}, context {BLOCK_SIZE}")
    print(f"Parameters:  {model.num_params():,}")
    print()
    shared["model"] = model                      # the model Step 7 trains
    shared["untrained"] = copy.deepcopy(model)   # an untouched copy, for before/after comparisons
    shared["rng_state"] = torch.get_rng_state()  # the random state training starts from

    for step in steps:
        if step == "overfit":
            step_overfit()
        else:
            STEPS[step]()


if __name__ == "__main__":
    main()
