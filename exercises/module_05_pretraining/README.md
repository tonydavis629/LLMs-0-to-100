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

From the `exercises/` folder:

```bash
uv run python module_05_pretraining/src/main.py
```

The runner goes through the steps in order. Each step's header line carries a tag, and the step's output follows: what your code produced, then one line per test. The tests live in `tests/`, one file per step. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 3: get_batch() === CORRECT
  One batch: x has shape (32, 128), y has shape (32, 128)
  x[0][:24] = ' guess who caused your f'
  y[0][:24] = 'guess who caused your fa'
  CORRECT    x and y both have shape (batch_size, block_size): (3, 4) here
  CORRECT    on data = 0, 1, ..., 99 every target is its input plus one (y = x + 1)
  CORRECT    shortest stream: 0..4 with block_size 4 gives x = [0,1,2,3], y = [1,2,3,4]
```

Run a single step with `--step` (1, 2, 3, 4, 5, 7, 8, or 10). Steps 1 and 2 always run first, because the other steps need the token streams they build:

```bash
uv run python module_05_pretraining/src/main.py --step 4
```

Step 7 runs the full pretraining loop once Steps 3 to 5 work: 2,000 optimizer steps, which take several minutes on a laptop CPU. It prints the train and validation loss every 250 steps and saves a loss-curve image to `output/`. Steps 8 and 10 then use the trained model. Step 8 reads its validation loss as perplexity and bits per token, and Step 10 prints a text sample from before and after training.

There is also a single-batch sanity check. It runs Steps 1 to 5, then trains on one small batch over and over in place of the full run. The loss should fall toward zero, which shows the model and optimizer can fit data and the loop is wired correctly:

```bash
uv run python module_05_pretraining/src/main.py --overfit
```

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. The learning-rate schedule (`lr_at_step()`) is provided in `src/schedules.py`. Run the finished answers with `--solution`:

```
uv run python module_05_pretraining/src/main.py --solution
```

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
| 7 | `estimate_loss()` | Average loss over several batches |
| 8 | `loss_to_perplexity_and_bits()` | `exp(loss)` and `loss / ln 2` |
| 10 | `generate()` | Sample one token at a time from the model |

Step 6 (the warmup + cosine learning-rate schedule, `lr_at_step()`) is provided in
full, and steps 9 (the loss-curve plot) and 11 (the overfit sanity check) are
handled by the runner using the functions above. The model (`src/model.py`), the
runner (`src/main.py`), and the plotting helper (`src/visualization.py`) are all
provided &mdash; you only edit `exercise.py`.

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
