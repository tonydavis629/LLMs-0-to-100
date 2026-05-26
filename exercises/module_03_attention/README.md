# Module 3: Attention Mechanisms

## Overview

Implement scaled dot-product attention from scratch on a tiny 5-token sequence. You will compute Q, K, V projections, raw attention scores, scaled softmax weights, and the weighted output. Then add a causal mask to see how it prevents tokens from looking ahead, and add positional embeddings to see how position information changes the attention pattern.

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

Output plots are saved to `module_03_attention/output/`. The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code (or two at most).

| Step | Function | What it does |
|------|----------|--------------|
| 1 | (provided) | Create token vectors for a tiny sequence |
| 2 | `compute_qkv()` | Project token embeddings into Q, K, V matrices |
| 3 | `raw_attention_scores()` | Compute pairwise compatibility: Q @ K^T |
| 4 | `scaled_softmax()` | Scale by 1/sqrt(d_k), then softmax |
| 5 | `attention_output()` | Weighted sum of value vectors |
| 6 | `causal_mask()` | Build a lower-triangular mask of 0s and -infs |
| 7 | (provided) | `masked_attention()` combines steps 3-6 with the mask |
| 8 | `add_positional_embeddings()` | Add position vectors to token embeddings |
| EC | `kv_cache_step()` | Simulate one-token-at-a-time generation with cached keys and values |

`src/main.py` is the runner and `src/visualization.py` holds the plotting helpers &mdash; both are provided. You should only need to edit `exercise.py`.

## Extra credit

Implement `kv_cache_step()` &mdash; simulate the KV cache used during autoregressive generation. Instead of recomputing keys and values for all tokens on every step, cache them and only compute the new key and value for the latest token. The runner processes tokens one at a time and compares generation cost with and without the cache. The full implementation lives in `solution/exercise.py`.
