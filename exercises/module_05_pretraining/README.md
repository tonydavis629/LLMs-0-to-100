# Module 5: Pretraining NanoGPT

## Overview

Train a tiny decoder-only language model **from scratch** on a bundled text file
(the public-domain "tiny Shakespeare" corpus). The model architecture is provided
&mdash; your job is the **pretraining loop** that turns random weights into useful
ones: encode text, build shifted `(input, target)` batches, compute cross-entropy
loss, take optimizer steps under a warmup + cosine learning-rate schedule, measure
validation loss as perplexity and bits per token, and sample text before and after
training.

The goal is not a useful model. The goal is to make the pretraining loop **visible
and measurable**: loss going down, validation loss tracking it, and samples
improving from random characters to text-like output.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_05_pretraining/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you can
fill in one step at a time and re-run immediately. It prints the model and dataset
size, a sample before training, the loss at each checkpoint, the final perplexity
and bits per token, a sample after training, and saves a loss-curve image to
`output/`.

There is also a single-batch sanity check:

```bash
uv run python exercises/module_05_pretraining/src/main.py --overfit
```

This trains repeatedly on one small batch; the loss should crater toward zero,
confirming the model and optimizer can fit data (i.e. the loop is wired correctly).

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each
needs only one expression or one short block.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `encode()` | Map characters to integer token IDs |
| 2 | `train_val_split()` | Split the token stream into train and validation |
| 3 | `get_batch()` | Build inputs `x` and **shifted** targets `y` |
| 4 | `compute_loss()` | Cross-entropy of logits against the next tokens |
| 5 | `train_step()` | Zero gradients, backpropagate (clip + step provided) |
| 6 | `lr_at_step()` | Cosine-decay coefficient (warmup provided) |
| 7 | `estimate_loss()` | Average loss over several batches |
| 8 | `perplexity_and_bits()` | `exp(loss)` and `loss / ln 2` |
| 10 | `generate()` | Sample one token at a time from the model |

Steps 9 (the loss-curve plot) and 11 (the overfit sanity check) are handled by the
runner using the functions above. The model (`src/model.py`), the runner
(`src/main.py`), and the plotting helper (`src/visualization.py`) are all provided
&mdash; you only edit `exercise.py`.

## Dataset

`data/tinyshakespeare.txt` is the public-domain concatenation of Shakespeare's
plays used by Karpathy's char-rnn / nanoGPT (~1.1 MB, 65 distinct characters). It
is character-level, so there is no tokenizer to train: the vocabulary is just the
set of characters, which keeps the focus on pretraining.

## Extra credit

- **Memorization vs generalization.** Make a tiny dataset with one phrase repeated
  many times, train, then check whether the model reproduces that exact phrase.
- **Prose vs code.** Swap the corpus for a file of source code and compare the
  generated samples (indentation, brackets, identifiers).
- **Gradient clipping.** The runner clips gradients inside `train_step`. Try raising
  the peak learning rate with clipping on vs off and watch for loss spikes.
- **Gradient accumulation.** Accumulate gradients over several micro-batches before
  each optimizer step and compare the effective batch size to the per-step one.
