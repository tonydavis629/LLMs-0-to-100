:::divider id="divider-exercise" title="Exercise" sub="Attention Visualization and Positional Embedding Visualization"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_03_attention/exercise.py` and fill in the `NotImplementedError` lines. Run after each step &mdash; unfinished functions are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Run the exercise (skips any not yet implemented)
cd exercises
uv run python module_03_attention/src/main.py
```

The exercise computes scaled dot-product attention step by step on a tiny 5-token sequence, then adds a causal mask and positional embeddings. Check the `output/` directory for plots after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Attention Mechanisms

Implement scaled dot-product attention on a 5-token sequence with $d_{\text{model}} = 8$ and $d_k = 4$. Each function is mostly written for you &mdash; you fill in **one key line**. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**Unmasked attention (steps 1-5)**

Compute Q, K, V, raw scores, softmax weights, and the weighted output.
+++
**Causal mask and position (steps 6-8)**

Add a causal mask to block future tokens, then add positional embeddings and observe the change in attention patterns.
:::

---

:::step id="exercise-step2-code" title="Step 2: compute_qkv()"
```python
def compute_qkv(X: torch.Tensor, d_k: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the token matrix X into query, key, and value matrices.

    Args:
        X: Token embeddings, shape (seq_len, d_model).
        d_k: Dimension of queries, keys, and values.

    Returns:
        (Q, K, V): Each of shape (seq_len, d_k).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    W_Q = torch.randn(d_model, d_k) * 0.1
    W_K = torch.randn(d_model, d_k) * 0.1
    W_V = torch.randn(d_model, d_k) * 0.1

    # TODO: Compute Q, K, V by projecting X through W_Q, W_K, W_V
    # HINT: use matrix multiplication X @ W_Q, X @ W_K, X @ W_V
    raise NotImplementedError("TODO: compute Q, K, V projections")
```
+++
**Hint:** Use matrix multiplication `X @ W_Q`, `X @ W_K`, and `X @ W_V` to project the token embeddings.
+++
**Answer:**

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
return Q, K, V
```
:::

---

:::step id="exercise-step3-code" title="Step 3: raw_attention_scores()"
```python
def raw_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise compatibility scores: Q @ K^T.

    Args:
        Q: Query matrix, shape (seq_len, d_k).
        K: Key matrix, shape (seq_len, d_k).

    Returns:
        Scores matrix, shape (seq_len, seq_len).
    """
    # TODO: Compute the attention scores as the matrix product of Q and K^T
    # HINT: use Q @ K.T (the .T attribute transposes a 2D tensor)
    raise NotImplementedError("TODO: compute raw attention scores Q @ K^T")
```
+++
**Hint:** Use `Q @ K.T` to compute the matrix product of queries and transposed keys.
+++
**Answer:**

```python
return Q @ K.T
```
:::

---

:::step id="exercise-step4-code" title="Step 4: scaled_softmax()"
```python
def scaled_softmax(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

    Args:
        scores: Raw attention scores, shape (seq_len, seq_len).
        d_k: Dimension of keys (used for scaling).

    Returns:
        Attention weights, shape (seq_len, seq_len). Each row sums to 1.
    """
    # TODO: Scale the scores by 1/sqrt(d_k), then apply softmax along dim=-1
    # HINT: divide scores by (d_k ** 0.5), then call F.softmax(..., dim=-1)
    raise NotImplementedError("TODO: apply scaled softmax to attention scores")
```
+++
**Hint:** Divide `scores` by `(d_k ** 0.5)`, then call `F.softmax(..., dim=-1)`.
+++
**Answer:**

```python
return F.softmax(scores / (d_k ** 0.5), dim=-1)
```
:::

---

:::step id="exercise-step5-code" title="Step 5: attention_output()"
```python
def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute the attention output as a weighted sum of value vectors.

    Args:
        weights: Attention weights, shape (seq_len, seq_len).
        V: Value matrix, shape (seq_len, d_k).

    Returns:
        Output tensor, shape (seq_len, d_k).
    """
    # TODO: Compute the weighted sum of values using the attention weights
    # HINT: use weights @ V (matrix multiply the weight matrix by the value matrix)
    raise NotImplementedError("TODO: compute attention output as weighted sum of values")
```
+++
**Hint:** Multiply the weight matrix by the value matrix: `weights @ V`.
+++
**Answer:**

```python
return weights @ V
```
:::

---

:::terminal id="exercise-step5-output" title="Steps 1&ndash;5: Unmasked Attention Output" cmd="uv run python module_03_attention/src/main.py" caption="Unmasked attention weights are nearly uniform because the random projections are small. Each row sums to 1.0."
<span class="header">============================================================
STEP 1: Create Token Vectors
============================================================</span>
<span class="success">Shape: torch.Size([5, 8])</span>

<span class="header">============================================================
STEP 2: Compute Q, K, V Projections
============================================================</span>
<span class="success">Q shape: torch.Size([5, 4])</span>
<span class="success">K shape: torch.Size([5, 4])</span>
<span class="success">V shape: torch.Size([5, 4])</span>

<span class="header">============================================================
STEP 3: Raw Attention Scores (QK^T)
============================================================</span>
<span class="success">Shape: torch.Size([5, 5])</span>

<span class="header">============================================================
STEP 4: Scaled Softmax Attention Weights
============================================================</span>
<span class="success">Row sums (should be ~1.0): [1.0, 1.0, 1.0, 1.0, 1.0]</span>

<span class="header">============================================================
STEP 5: Attention Output (Weighted Sum of Values)
============================================================</span>
<span class="success">Shape: torch.Size([5, 4])</span>
:::

---

:::step id="exercise-step6-code" title="Step 6: causal_mask()"
```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower-triangular) mask for autoregressive attention.

    A causal mask prevents each token from attending to future tokens.
    Entry (i, j) is 0 if j <= i (token i may attend to token j),
    and -inf if j > i (token i must NOT attend to token j).

    Args:
        seq_len: Length of the sequence.

    Returns:
        Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
    """
    # TODO: Create a lower-triangular mask of 0s with -inf above the diagonal
    # HINT: use torch.tril(torch.ones(seq_len, seq_len)) for the lower triangle,
    #        then convert: 1 -> 0.0 and 0 -> float('-inf')
    raise NotImplementedError("TODO: create causal mask")
```
+++
**Hint:** Start with `torch.tril(torch.ones(seq_len, seq_len))`, then use `.masked_fill(mask == 0, float('-inf'))` to replace zeros with $-\infty$.
+++
**Answer:**

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
mask = mask.masked_fill(mask == 0, float('-inf'))
return mask
```
:::

---

:::step id="exercise-step8-code" title="Step 8: add_positional_embeddings()"
```python
def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add learned positional embeddings to the token representations.

    Args:
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        X_pos: Token embeddings plus positional embeddings, shape (seq_len, d_model).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    P = torch.randn(seq_len, d_model) * 0.1

    # TODO: Add the positional embeddings P to the token embeddings X
    # HINT: simply add X + P (elementwise addition of the two tensors)
    raise NotImplementedError("TODO: add positional embeddings to token embeddings")
```
+++
**Hint:** Simply add the two tensors: `X + P`.
+++
**Answer:**

```python
return X + P
```
:::

---

:::terminal id="exercise-step8-output" title="Steps 6&ndash;8: Causal Mask and Positional Embeddings" cmd="uv run python module_03_attention/src/main.py" caption="The causal mask zeroes out the upper triangle, forcing each token to attend only to itself and earlier tokens."
<span class="header">============================================================
STEP 6: Causal Mask
============================================================</span>
<span class="success">Mask matrix:
[[  1. -inf -inf -inf -inf]
 [  1.   1. -inf -inf -inf]
 [  1.   1.   1. -inf -inf]
 [  1.   1.   1.   1. -inf]
 [  1.   1.   1.   1.   1.]]</span>

<span class="header">============================================================
STEP 7: Masked Attention Output
============================================================</span>
<span class="success">Masked weight matrix:
[[1.    0.    0.    0.    0.   ]
 [0.501 0.499 0.    0.    0.   ]
 [0.333 0.334 0.334 0.    0.   ]
 [0.25  0.251 0.25  0.25  0.   ]
 [0.2   0.2   0.2   0.2   0.2  ]]</span>

<span class="header">============================================================
STEP 8: Positional Embeddings
============================================================</span>
<span class="success">Shape: torch.Size([5, 8])</span>

<span class="header">============================================================
VISUALIZATIONS
============================================================</span>
<span class="success">Saved attention comparison to output/attention_comparison.png</span>
<span class="success">Saved positional effect to output/positional_effect.png</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: KV Cache

Implement `kv_cache_step()` &mdash; simulate the KV cache used during autoregressive generation. Instead of recomputing keys and values for all tokens, cache them and only project the new token. <!-- .element: class="text-lg" -->

```python
def kv_cache_step(
    new_token: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
    d_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: project new_token through W_Q, W_K, W_V, then append K and V to cache
    new_key = None
    new_value = None
    if new_key is None or new_value is None:
        raise NotImplementedError("Extra credit: project new_token through W_Q, W_K, W_V, then append K and V to cache")
```

The runner processes tokens one at a time and compares generation cost with and without the cache. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
