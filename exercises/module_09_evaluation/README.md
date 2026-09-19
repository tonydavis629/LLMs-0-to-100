# Module 9: Build a Small Benchmark Suite and Score Two Models

## Overview

Nothing here trains a model. Two finished checkpoints are bundled &mdash; the
**instruct model** from Module 6 and the **RL model** Module 7 produced by running
GRPO on the reverse task &mdash; and your job is to write the **metrics** that decide
which one is better.

You implement the eight scoring functions the lecture describes: perplexity, answer
normalization, exact match, token-level F1, likelihood-scored multiple choice,
pass@k, per-task accuracy, and the suite average. The runner then applies them to
both checkpoints and prints the report.

The payoff is the per-task table. The RL model wins the task it was trained on by a
wide margin. The overall average moves much less than that, and one task moves the
other way. Both facts are in the same run, and only one of them survives into a
headline number.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```
cd exercises
uv run python module_09_evaluation/src/main.py
```

The runner first prints the **protocol** (models, chat template, normalization,
generation budget, decoding settings, seed, suite size), because a score cannot be
reproduced without it. Then it goes through the steps in order. Each step's header
line carries a tag, and the step's output follows: what your metric reports about
the two models, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 4: token_f1() === CORRECT
  Greedy answers that earn partial credit (F1 between 0 and 1):
    [qa] 'opposite of day?'   want 'it is night'
        rl       'it is cold'                 0.67
  CORRECT    partial credit: 'it is bluu' against 'it is blue' has P = R = 2/3, so F1 = 0.667
  CORRECT    P and R differ: 'blue' against 'it is blue' has P = 1, R = 1/3, so F1 = 0.5
  CORRECT    a correct answer scores 1.0: 'It is blue.' against 'it is blue'
```

Run a single step with `--step` (1 to 8). A single-step run skips the protocol:

```
uv run python module_09_evaluation/src/main.py --step 4
```

The tests live in `tests/`, one file per step. Each calls your function on small
inputs whose correct answer is known. Steps 3 to 8 share one generation pass (50
greedy and 250 sampled answers per model), so the first of them to run takes a few
seconds longer. Step 6 saves a grouped bar chart to `output/task_comparison.png`.

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. Run the finished answers with `--solution`:

```
uv run python module_09_evaluation/src/main.py --solution
```

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each needs
only one expression or one short line.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `perplexity()` | Average token loss &rarr; perplexity |
| 2 | `normalize_answer()` | Lowercase, strip punctuation, collapse whitespace |
| 3 | `exact_match()` | 1.0 if the normalized answer matches any acceptable answer |
| 4 | `token_f1()` | Harmonic mean of token precision and recall |
| 5 | `task_accuracy()` | Mean score within each task |
| 6 | `suite_score()` | Mean of the per-task scores |
| 7 | `score_multiple_choice()` | Pick the option with the best per-token log-probability |
| 8 | `pass_at_k()` | `1 - C(n-c, k) / C(n, k)` from `n` samples and `c` correct |

Steps 1&ndash;6 are enough to produce the full per-task report. Steps 7 and 8 add the
two benchmarks that tell a different story about the same models.

The model (`src/model.py`), tokenizer (`src/tokenizer.py`), evaluation data
(`src/data.py`), plotting (`src/visualization.py`), and runner (`src/main.py`) are
all provided. You only edit `exercise.py`.

## Data

- `data/instruct_model.pt` &mdash; the Module 6 story: the Module 5 base model,
  vocabulary widened for the four chat special tokens, supervised-finetuned on four
  toy tasks (`uppercase`, `repeat`, `qa`, `reverse`).
- `data/rl_model.pt` &mdash; the Module 7 story: that same checkpoint after GRPO on
  the **reverse task only**, with an exact-match verifiable reward.
  (Both checkpoints were trained for this module by `src/make_checkpoints.py`,
  starting from `../module_06_finetuning/data/base_model.pt`; regenerate them with that
  script.)
- `data/tasks.jsonl` &mdash; 50 held-out cases, each `{id, task, prompt, answers}`.
  `answers` is a list because more than one string can be correct.
- `data/multiple_choice.jsonl` &mdash; 16 four-option questions, each
  `{id, question, options, answer_index}`, scored by likelihood with no generation.
- `data/held_out.txt` &mdash; a slice of the Module 5 corpus from the validation
  region, used only for perplexity.

Both checkpoints are regenerated by `src/make_checkpoints.py`, and the
evaluation files by `src/data.py`.

A deliberate trap is built into the data: the `uppercase`, `repeat`, and `reverse`
eval cases use words that appear nowhere in finetuning, but the eight `qa` facts the
suite tests are exactly the eight facts the models memorized. That task is
**contaminated by construction**, and its score should be read as recall, not
generalization. The contamination extra credit makes this visible.

## Extra credit

- **Prompt sensitivity.** Change the chat template in `_prefix_ids()` (add a space
  after the assistant marker, or reword the prompts) and re-run. Report how far the
  scores move on identical weights &mdash; this is the protocol dependence from
  section g, measured rather than asserted.
- **Contamination check.** Search the finetuning pairs from `src/data.py` for each
  evaluation prompt and count the exact overlaps. Every `qa` case will hit. Then
  re-run the suite with `qa` excluded and see what happens to the headline number.
- **Length normalization.** Score the multiple-choice set with **total**
  log-probability instead of the per-token average. On this set both models still
  get 16/16. Print each option's token count and score to explain why, then add a
  question whose wrong option is much shorter than the right one and see whether the
  two methods still agree.
- **Bootstrap confidence interval.** Resample the 50 cases with replacement 1000
  times, recompute the accuracy difference between the two models each time, and
  report the 5th and 95th percentiles. Decide whether the gap on any single task is
  larger than the noise.
- **Judge order bias.** Write a rule-based judge that prefers the longer answer,
  score the two models' answers as pairs in both orders, and count how often the
  preference flips. This is the position-bias demonstration from section c with a
  judge simple enough to read.
- **pass@k curve.** Raise `N_SAMPLES` to 20 and plot pass@k for k = 1, 2, 5, 10, 20
  for both models on the reverse task. Check whether the RL model's advantage
  shrinks as k grows &mdash; the Module 7 claim that RL sharpens rather than expands.
