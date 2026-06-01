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

The exercise implements a tiny attention layer on a 5-token sequence, then adds a causal mask and sinusoidal positional encodings. Check the `output/` directory for plots after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Attention Mechanisms

Implement scaled dot-product attention inside `TinyAttentionLayer` with $d_{\text{model}} = 8$ and $d_k = 4$. The layer owns the random projection weights; you fill in the key tensor operations. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**Unmasked attention (steps 1-5)**

Compute Q, K, V from the layer weights, then compute raw scores, softmax weights, and the weighted output.
+++
**Causal mask and position (steps 6-8)**

Add a causal mask to block future tokens, then add sinusoidal positional encodings and observe the change in attention patterns.
:::

---

:::step id="exercise-step2-code" title="Step 2: TinyAttentionLayer.compute_qkv()"
```python
    def compute_qkv(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project the token matrix X into query, key, and value matrices.

        Args:
            X: Token embeddings, shape (seq_len, d_model).

        Returns:
            (Q, K, V): Each tensor has shape (seq_len, d_k).
        """
        # TODO: Compute and return Q, K, V by multiplying X by this layer's W_Q, W_K, and W_V.
        raise NotImplementedError("TODO: compute Q, K, V projections")
```
+++
**Hint:** use X @ self.W_Q, X @ self.W_K, and X @ self.W_V.
+++
**Answer:**

```python
return X @ self.W_Q, X @ self.W_K, X @ self.W_V
```
:::

---

:::step id="exercise-step3-code" title="Step 3: raw_attention_scores()"
```python
    def raw_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise compatibility scores: Q @ K^T.

        Args:
            Q: Query matrix, shape (seq_len, d_k).
            K: Key matrix, shape (seq_len, d_k).

        Returns:
            Scores matrix, shape (seq_len, seq_len).
        """
        # TODO: Compute the attention scores as the matrix product of Q and K^T.
        raise NotImplementedError("TODO: compute raw attention scores Q @ K^T")
```
+++
**Hint:** use Q @ K.T (the .T attribute transposes a 2D tensor).
+++
**Answer:**

```python
return Q @ K.T
```
:::

---

:::step id="exercise-step4-code" title="Step 4: scaled_softmax()"
```python
    def scaled_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

        Args:
            scores: Raw attention scores, shape (seq_len, seq_len).

        Returns:
            Attention weights, shape (seq_len, seq_len). Each row sums to 1.
        """
        # TODO: Scale the scores by 1/sqrt(d_k), then apply softmax along dim=-1.
        raise NotImplementedError("TODO: apply scaled softmax to attention scores")
```
+++
**Hint:** divide scores by (self.d_k ** 0.5), then call F.softmax(..., dim=-1).
+++
**Answer:**

```python
return F.softmax(scores / (self.d_k ** 0.5), dim=-1)
```
:::

---

:::step id="exercise-step5-code" title="Step 5: attention_output()"
```python
    def attention_output(self, weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute the attention output as a weighted sum of value vectors.

        Args:
            weights: Attention weights, shape (seq_len, seq_len).
            V: Value matrix, shape (seq_len, d_k).

        Returns:
            Output tensor, shape (seq_len, d_k).
        """
        # TODO: Compute the weighted sum of values using the attention weights.
        raise NotImplementedError("TODO: compute attention output as weighted sum of values")
```
+++
**Hint:** use weights @ V (matrix multiply the weight matrix by the value matrix).
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
<span class="success">First output vector: [-0.006  0.002  0.027  0.019]</span>
:::

---

:::step id="exercise-step6-code" title="Step 6: causal_mask()"
```python
    def causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal mask for autoregressive attention.

        Entry (i, j) is 0 if j <= i, which means token i may attend to token j.
        Entry (i, j) is -inf if j > i, which blocks future tokens before softmax.

        Args:
            seq_len: Length of the sequence.

        Returns:
            Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
        """
        # Create a lower-triangular matrix: 1 means allowed, 0 means blocked.
        allowed = torch.tril(torch.ones(seq_len, seq_len))
        # TODO: Convert allowed positions to 0.0 and blocked positions to -inf.
        raise NotImplementedError("TODO: create causal mask")
```
+++
**Hint:** use masked_fill twice: first replace 0s with -inf, then replace 1s with 0.0.
+++
**Answer:**

```python
mask = allowed.masked_fill(allowed == 0, float("-inf"))
return mask.masked_fill(mask == 1, 0.0)
```
:::

---

:::step id="exercise-step8-code" title="Step 8: add_positional_embeddings()"
```python
def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add sinusoidal positional encodings to the token representations."""
    seq_len, d_model = X.shape
    position = torch.arange(seq_len, dtype=X.dtype, device=X.device).unsqueeze(1)
    dim_pair = torch.arange(0, d_model, 2, dtype=X.dtype, device=X.device)
    angle_rates = 1 / (10000 ** (dim_pair / d_model))
    angles = position * angle_rates
    P = torch.zeros_like(X)
    P[:, 0::2] = torch.sin(angles)

    # TODO: Fill the odd dimensions of P with cosine values from the sinusoidal equation.
    raise NotImplementedError("TODO: fill cosine positional dimensions")

    return X + P
```
+++
**Hint:** use torch.cos(angles) and assign the result to P[:, 1::2].
+++
**Answer:**

```python
P[:, 1::2] = torch.cos(angles)
return X + P
```
:::

---

:::terminal id="exercise-step8-output" title="Steps 6&ndash;8: Causal Mask and Positional Embeddings" cmd="uv run python module_03_attention/src/main.py" caption="The causal mask zeroes out the upper triangle, forcing each token to attend only to itself and earlier tokens."
<span class="header">============================================================
STEP 6: Causal Mask
============================================================</span>
<span class="success">Mask first row: [0. -inf -inf -inf -inf]</span>
<span class="success">Mask last row:  [0. 0. 0. 0. 0.]</span>

<span class="header">============================================================
STEP 7: Masked Attention Output
============================================================</span>
<span class="success">Masked first row: [1. 0. 0. 0. 0.]</span>
<span class="success">Masked last row:  [0.2 0.2 0.2 0.2 0.2]</span>

<span class="header">============================================================
STEP 8: Positional Embeddings
============================================================</span>
<span class="success">Shape: torch.Size([5, 8])</span>
<span class="success">First token (after pos):  [ 0.164  0.984 -0.05   1.044 -0.076  1.108  0.08   1.168]</span>
<span class="success">Attention weights WITH position: first row [0.198 0.2   0.203 0.202 0.197]</span>

<span class="header">============================================================
VISUALIZATIONS
============================================================</span>
<span class="success">Saved attention comparison to module_03_attention/output/attention_comparison.png</span>
<span class="success">Saved positional effect to module_03_attention/output/positional_effect.png</span>
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
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    d_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: Project new_token through W_Q, W_K, and W_V.
    new_query = None
    new_key = None
    new_value = None
    if new_query is None or new_key is None or new_value is None:
        raise NotImplementedError("Extra credit: project new_token through W_Q, W_K, and W_V")
```

The runner processes tokens one at a time and compares generation cost with and without the cache. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
