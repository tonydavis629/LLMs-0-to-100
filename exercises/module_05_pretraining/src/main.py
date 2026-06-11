"""
Module 5 Exercise runner: Pretraining NanoGPT

Run with:
    uv run python module_05_pretraining/src/main.py
    uv run python module_05_pretraining/src/main.py --overfit   # single-batch sanity check

Trains a tiny decoder-only language model from scratch on a bundled text file.
The goal is not a useful model; it is to make the pretraining loop visible:
loss going down, validation loss tracking it, perplexity and bits per token
falling, and samples improving from random characters to text-like output.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one step at a time and re-run immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable so we can grab the provided model + visualization helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    encode,
    train_val_split,
    get_batch,
    compute_loss,
    train_step,
    lr_at_step,
    estimate_loss,
    perplexity_and_bits,
    generate,
)
from model import GPTConfig, TinyGPT  # noqa: E402
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

_THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _THIS_DIR.parent / "output"


def _find_data_file() -> Path:
    """Walk up from this file to find data/tinyshakespeare.txt (works from src/ or solution/src/)."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / "tinyshakespeare.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate data/tinyshakespeare.txt")


def _is_implemented(fn, *args, **kwargs) -> bool:
    """Return True if fn runs without raising NotImplementedError (errors aside)."""
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception still means the student wrote *something*; treat as implemented.
        return True


def _probe_steps() -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs.

    Uses a tiny scratch model so probing train_step / estimate_loss / generate
    never disturbs the real model that we are about to train.
    """
    scratch = TinyGPT(GPTConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=16))
    opt = torch.optim.AdamW(scratch.parameters(), lr=1e-3)
    data = torch.randint(0, 7, (64,))
    x = torch.randint(0, 7, (2, 8))
    y = torch.randint(0, 7, (2, 8))
    logits = torch.randn(2, 8, 7)
    return {
        "encode": _is_implemented(encode, "ab", {"a": 0, "b": 1}),
        "split": _is_implemented(train_val_split, torch.arange(10)),
        "get_batch": _is_implemented(get_batch, data, 8, 2),
        "compute_loss": _is_implemented(compute_loss, logits, y),
        "train_step": _is_implemented(train_step, scratch, opt, x, y),
        "lr": _is_implemented(lr_at_step, 0, 10, 100, 1e-3, 1e-4),
        "estimate_loss": _is_implemented(estimate_loss, scratch, data, 8, 2, 1),
        "perplexity": _is_implemented(perplexity_and_bits, 1.5),
        "generate": _is_implemented(generate, scratch, torch.zeros(1, 1, dtype=torch.long), 2, 8),
    }


def _heading(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _decode(itos: dict[int, str], ids: torch.Tensor) -> str:
    return "".join(itos[int(i)] for i in ids)


def _sample(model, itos, stoi, generator) -> str:
    """Generate SAMPLE_TOKENS characters from the seed text (uses student generate)."""
    seed_ids = torch.tensor([[stoi[c] for c in SAMPLE_SEED_TEXT]], dtype=torch.long)
    out = generate(model, seed_ids, SAMPLE_TOKENS, BLOCK_SIZE, temperature=1.0, generator=generator)
    return _decode(itos, out[0])


def run_overfit(model, train_data, steps: int = 300) -> None:
    """Single-batch overfit sanity check: loss on one fixed batch should fall toward 0."""
    _heading("OVERFIT ONE BATCH (sanity check)")
    gen = torch.Generator().manual_seed(SEED)
    # A small fixed batch the model can memorize, so the loss should crater to ~0.
    x, y = get_batch(train_data, BLOCK_SIZE, OVERFIT_BATCH_SIZE, gen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    print(f"Training repeatedly on ONE batch of shape {tuple(x.shape)} for {steps} steps:")
    print(f"{'step':>6}  {'loss':>8}")
    for step in range(steps + 1):
        loss = train_step(model, optimizer, x, y, GRAD_CLIP)
        if step % 50 == 0:
            print(f"{step:>6}  {loss:>8.4f}")
    print("\nIf the loss approaches 0, the model and optimizer can fit data: the loop works.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a tiny GPT.")
    parser.add_argument("--overfit", action="store_true",
                        help="run the single-batch overfit sanity check instead of full training")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Data: load the corpus and build a character-level vocabulary.
    # (No student code needed for this part.)
    # ------------------------------------------------------------------
    _heading("MODULE 5: Pretraining NanoGPT")
    data_file = _find_data_file()
    text = data_file.read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"Corpus:      {data_file.name}  ({len(text):,} characters)")
    print(f"Vocabulary:  {vocab_size} unique characters")
    print()

    steps_done = _probe_steps()

    # ------------------------------------------------------------------
    # Build the model (provided) and report its size.
    # ------------------------------------------------------------------
    cfg = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT)
    model = TinyGPT(cfg)
    _heading("MODEL")
    print(f"TinyGPT: {N_LAYER} layers, {N_HEAD} heads, width {N_EMBD}, context {BLOCK_SIZE}")
    print(f"Parameters:  {model.num_params():,}")
    print()

    # ------------------------------------------------------------------
    # Step 1: encode the text into token IDs.
    # ------------------------------------------------------------------
    _heading("STEP 1: Encode text into token IDs")
    if not steps_done["encode"]:
        print("  [skipped: implement encode()]  -- nothing downstream can run yet\n")
        return
    data = encode(text, stoi)
    print(f"  Encoded {len(data):,} tokens. First 20 IDs: {data[:20].tolist()}")
    print()

    # ------------------------------------------------------------------
    # Step 2: train/validation split.
    # ------------------------------------------------------------------
    _heading("STEP 2: Train / validation split")
    if not steps_done["split"]:
        print("  [skipped: implement train_val_split()]\n")
        return
    train_data, val_data = train_val_split(data, VAL_FRACTION)
    print(f"  Train tokens: {len(train_data):,}   Validation tokens: {len(val_data):,}")
    print()

    # ------------------------------------------------------------------
    # Sample BEFORE training (needs Step 10: generate).
    # ------------------------------------------------------------------
    _heading("SAMPLE BEFORE TRAINING (random weights)")
    gen = torch.Generator().manual_seed(SEED)
    if steps_done["generate"]:
        before = _sample(model, itos, stoi, gen)
        print(repr(before))
    else:
        print("  [skipped: implement generate()]")
    print()

    # ------------------------------------------------------------------
    # The training loop needs Steps 3-7. Check them together.
    # ------------------------------------------------------------------
    needed = ["get_batch", "compute_loss", "train_step", "lr", "estimate_loss"]
    missing = [s for s in needed if not steps_done[s]]
    ckpt_steps: list[int] = []
    train_hist: list[float] = []
    val_hist: list[float] = []
    final_val = None

    if args.overfit:
        if missing:
            _heading("OVERFIT ONE BATCH (sanity check)")
            print(f"  [skipped: implement {', '.join(missing)} first]\n")
        else:
            run_overfit(model, train_data)
        return

    _heading("TRAINING")
    if missing:
        print(f"  [skipped: implement {', '.join(missing)} to train]\n")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        batch_gen = torch.Generator().manual_seed(SEED)
        eval_gen = torch.Generator().manual_seed(SEED + 1)
        print(f"{'step':>6}  {'lr':>9}  {'train':>8}  {'val':>8}")
        for step in range(MAX_STEPS + 1):
            # Set this step's learning rate from the schedule (Step 6).
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
                final_val = va
                print(f"{step:>6}  {lr:>9.2e}  {tr:>8.4f}  {va:>8.4f}")

            if step == MAX_STEPS:
                break

            # One optimizer step on a fresh batch (Steps 3 + 4 + 5).
            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, batch_gen)
            train_step(model, optimizer, x, y, GRAD_CLIP)
        print()

    # ------------------------------------------------------------------
    # Step 8: perplexity and bits per token from the final validation loss.
    # ------------------------------------------------------------------
    _heading("STEP 8: Perplexity and bits per token")
    if final_val is not None and steps_done["perplexity"]:
        ppl, bpt = perplexity_and_bits(final_val)
        print(f"  Final validation loss: {final_val:.4f} nats")
        print(f"  Perplexity (exp loss): {ppl:.2f}   (effective next-char choices)")
        print(f"  Bits per token (loss / ln 2): {bpt:.4f}")
    elif final_val is None:
        print("  [skipped: training did not run]")
    else:
        print("  [skipped: implement perplexity_and_bits()]")
    print()

    # ------------------------------------------------------------------
    # Sample AFTER training.
    # ------------------------------------------------------------------
    _heading("SAMPLE AFTER TRAINING")
    if steps_done["generate"] and final_val is not None:
        after = _sample(model, itos, stoi, gen)
        print(repr(after))
    elif not steps_done["generate"]:
        print("  [skipped: implement generate()]")
    else:
        print("  [skipped: training did not run]")
    print()

    # ------------------------------------------------------------------
    # Step 9: plot the loss curve from the recorded checkpoints (provided).
    # ------------------------------------------------------------------
    _heading("STEP 9: Loss curve")
    if len(ckpt_steps) >= 2:
        plot_loss_curve(ckpt_steps, train_hist, val_hist, str(OUTPUT_DIR / "loss_curve.png"))
    else:
        print("  [skipped: no training checkpoints to plot]")
    print()

    _heading("Done")
    print("Try the single-batch sanity check:  uv run python module_05_pretraining/src/main.py --overfit")


if __name__ == "__main__":
    main()
