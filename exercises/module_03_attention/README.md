# Module 3: Attention Mechanisms

## Overview

Implement scaled dot-product attention from scratch on a tiny 5-token sequence. You will work inside a small `TinyAttentionLayer` class, compute Q, K, V projections from the layer's stored weight matrices, produce attention weights, and compute the weighted output. Then add a causal mask and sinusoidal positional encodings.

We use PyTorch tensors throughout, but every step is explicit — nothing is hidden behind `nn.MultiheadAttention`.

## Setup

From the `exercises/` directory:

```
uv sync
```

## Running

```
uv run python module_03_attention/src/main.py
```

The runner goes through the steps in order. Each step's header line carries a tag, and the step's output follows: the matrices your code produced for the 5-token sentence, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 6: causal_mask() === CORRECT
  Mask for 5 tokens, row = query, column = key:
            the     cat     sat      on     mat
   the        0    -inf    -inf    -inf    -inf
   cat        0       0    -inf    -inf    -inf
   sat        0       0       0    -inf    -inf
    on        0       0       0       0    -inf
   mat        0       0       0       0       0
  CORRECT    seq_len=3 gives [[0,-inf,-inf], [0,0,-inf], [0,0,0]]
  CORRECT    after softmax, 4 equal scores give row i weight 1/(i+1) on tokens 0..i and exactly 0 after
  CORRECT    used as attn_mask on 5 tokens, matches torch's is_causal=True attention
```

Some steps print results that need an earlier function. Step 4, for example, normalizes the scores from Steps 2 and 3. If that earlier function is unfinished, the step stays `INCOMPLETE` and names the step it is waiting on. Run a single step with `--step` (1 to 8, or `ec`):

```
uv run python module_03_attention/src/main.py --step 4
```

Steps 7 and 8 and the extra credit save plots to `module_03_attention/output/`. `exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`; `make_token_vectors()` is provided in `src/embeddings.py`. Run the finished answers with `--solution`:

```
uv run python module_03_attention/src/main.py --solution
```

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code (or two at most).

| Step | Function | What it does |
|------|----------|--------------|
| 1 | (provided) | Create token vectors for a tiny sequence |
| 2 | `TinyAttentionLayer.compute_qkv()` | Project token embeddings into Q, K, V matrices |
| 3 | `TinyAttentionLayer.raw_attention_scores()` | Compute pairwise compatibility: Q @ K^T |
| 4 | `TinyAttentionLayer.scaled_softmax()` | Scale by 1/sqrt(d_k), then softmax |
| 5 | `TinyAttentionLayer.attention_output()` | Weighted sum of value vectors |
| 6 | `TinyAttentionLayer.causal_mask()` | Build a lower-triangular mask of 0s and -infs |
| 7 | (provided) | `TinyAttentionLayer.masked_attention()` combines the previous steps with the mask |
| 8 | `add_positional_embeddings()` | Add sinusoidal positional encodings |
| EC | `kv_cache_step()` | Simulate one-token-at-a-time generation with cached keys and values |

`src/main.py` is the runner and `src/visualization.py` holds the plotting helpers. Both are provided. The tests live in `tests/`, one file per step (`test_step2_qkv.py` through `test_step8_positional.py`, plus `test_extra_credit.py`). Each one calls your function on small tensors with a known answer, and most also compare your result with PyTorch's own `F.scaled_dot_product_attention`. Read the test for the step you are on to see exactly what is expected. Step 1 has no tests because nothing in it is yours to write. Step 7 is also provided, but its tests check that your Steps 2, 3, 5, and 6 work together. You should only need to edit `exercise.py`.

## Extra credit

Implement `kv_cache_step()` to simulate the KV cache used during autoregressive generation. Instead of recomputing keys and values for all tokens on every step, cache them and only compute the new key and value for the latest token. The runner feeds the five tokens in one at a time, starting from an empty cache, and plots generation cost with and without the cache. Its tests check the empty-cache case by hand, confirm that each call adds one key and one value to the cache, and compare a token-by-token run with causal attention computed over the whole sequence at once. The full implementation lives in `solution/exercise.py`.
