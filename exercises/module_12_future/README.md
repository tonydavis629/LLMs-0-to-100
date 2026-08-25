# Module 12: Linear Attention, Two Ways

## Overview

Nothing here trains a model, and there is no dataset. The inputs are random
tensors from a fixed seed, because the thing being measured is a property of the
arithmetic rather than of any text.

You implement linear attention twice. The **parallel form** builds the same
n-by-n score matrix you wrote in Module 3, minus the softmax, and costs O(n^2).
The **recurrent form** carries a fixed-size state forward one token at a time,
exactly like the RNNs the transformer replaced in Module 4, and costs O(n). Then
you check whether the two agree.

They do. Katharopoulos et al. (2020) titled their paper "Transformers are RNNs"
because dropping the softmax makes attention factorize, and a factorized
attention runs either way. The exercise turns that sentence into something you
have checked yourself, then times both forms so the complexity claim stops being
a claim.

The last line of the timing table is the one worth staring at. At short
sequences the recurrent form is roughly ten times slower than either parallel
form, and by 8192 tokens it is the fastest of the three. Linear beats quadratic
eventually rather than immediately, which is easy to forget when reading a
complexity bound.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_12_future/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you
can fill in one function at a time and re-run immediately. The parallel form
comes alive after step 3, the recurrent form joins after step 5, the equivalence
check after step 6, and the timing sweep after step 7.

It prints the equivalence check first, then a timing table across sequence
lengths from 512 to 8192, then the fitted exponent of each cost curve, and saves
a log-log plot to `output/attention_scaling.png`.

## What you edit

Only `exercise.py`, at the module root. Each of the seven blanks is one line or
one short expression, marked with a `# TODO` describing what to return and a
`# HINT` pointing at the operation to use.

Everything in `src/` is provided plumbing: the softmax attention baseline, the
loop that drives your per-token functions across a sequence, the random input
generator, and the plotting.

## The steps

1. `feature_map`: the positive feature map `elu(x) + 1` that replaces the exponential
2. `masked_scores`: the n-by-n score matrix, causally masked by multiplying by zero
3. `parallel_linear_attention`: normalize the scores and multiply by the values
4. `update_state`: absorb one token into the running state `S` and normalizer `z`
5. `recurrent_step_output`: read one token's output back out of the state
6. `outputs_match`: check the two forms agree, up to floating-point noise
7. `time_forward`: measure one forward pass so the cost curves can be plotted

## Extra credit

- **Gated recurrence.** Multiply `S` by a decay factor (say 0.99) before each
  update and watch outputs start favoring recent tokens. You are now one scalar
  away from the forgetting mechanisms in RWKV and Mamba.
- **Memory accounting.** Compute the KV cache size in bytes at each sequence
  length using Module 10's formula, and print it beside the recurrent state's
  constant size. The table is more persuasive than the runtime plot.
- **Sharpness.** Compute the entropy of each row of the softmax attention
  weights and of the normalized linear attention weights on the same inputs.
  The linear kernel's blur is the recall cost, made visible without training
  anything.
- **Fit the exponents yourself.** The runner fits log time against log n by
  least squares. Do it by hand on the printed table and confirm you get the same
  numbers. This is the same log-log fit the scaling-law literature uses on loss.
- **One gradient step.** Rewrite the state update as a single step of gradient
  descent on a squared reconstruction loss for a linear map from keys to values,
  and confirm it reproduces the same `S` as step 4. The recurrent state is not
  just memory; it is a tiny model being trained during the forward pass. This is
  where the lecture's test-time training section starts.

## Solution

A complete reference implementation, including the runner's output and the
generated figure, is in `solution/`.
