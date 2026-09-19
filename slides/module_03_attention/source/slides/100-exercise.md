:::divider id="divider-exercise" title="Exercise" sub="Attention Visualization and Positional Embedding Visualization"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_03_attention/exercise.py`, the only file you edit, and fill in the `NotImplementedError` lines. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_03_attention/src/main.py

# Run a single step (1-8, or ec for the extra credit)
uv run python module_03_attention/src/main.py --step 4
```

You build a tiny attention layer on a 5-token sequence, then add a causal mask and sinusoidal positional encodings. Plots land in `output/` after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints the matrices your code produces for the 5-token sentence, then runs the step's **tests** from `tests/`. Each test calls your function on small tensors whose correct answer is known, and most also compare your result with PyTorch's `F.scaled_dot_product_attention`. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or the step needs an earlier function you have not finished (it names that step)

The tag sits on the step's header line, and the matrices and individual test results follow it. A 5 by 5 grid of numbers near 0.2 is hard to judge by eye, so rely on the tests to tell you when a step is done. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_03_attention/src/main.py --step 4" maxw="960px" caption="Here <code>scaled_softmax()</code> skipped the division by &radic;d<sub>k</sub>. The weights still sum to 1, so that test passes. The hand-computed test expected 0.731, got 0.881, and asks whether you divided by sqrt(d_k)."
<span class="header">=== Step 4: scaled_softmax() ===</span> <span class="t-fail">INCORRECT</span>
  Weights (5, 5), row = query, column = key:
            the     cat     sat      on     mat
   the   0.2001  0.2000  0.2000  0.1999  0.1999
   cat   0.2006  0.1989  0.1996  0.2006  0.2004
   sat   0.1996  0.2003  0.2003  0.1999  0.1999
    on   0.1994  0.2009  0.1999  0.1999  0.1999
   mat   0.1996  0.2006  0.2001  0.1998  0.1999
  Row sums: [1.0, 1.0, 1.0, 1.0, 1.0]
  <span class="t-fail">INCORRECT</span>  d_k=4 divides by 2: scores [[2,0],[0,0]] give weights [[0.731,0.269],[0.5,0.5]]
             expected [[0.7311, 0.2689], [0.5, 0.5]]
             got      [[0.8808, 0.1192], [0.5, 0.5]] (did you divide by sqrt(d_k) before the softmax?)
  <span class="success">CORRECT</span>    each query's weights over the keys sum to 1 (3 queries x 5 keys)
  <span class="t-fail">INCORRECT</span>  matches the weights inside torch's F.scaled_dot_product_attention (4 tokens, d_k=4)
             expected row 0 = [0.2441, 0.2207, 0.3027, 0.2325]
             got      row 0 = [0.2346, 0.1918, 0.3608, 0.2128]
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Attention Mechanisms

Implement scaled dot-product attention inside `TinyAttentionLayer` ($d_{\text{model}} = 8$, $d_k = 4$). The layer owns the random projection weights; you fill in the tensor operations, and every step has its own tests in `tests/`. <!-- .element: class="text-lg" -->

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
**Hint:** matrix-multiply `X` by each of the layer's three weight matrices with the `@` operator.
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
**Hint:** matrix-multiply `Q` by the transpose of `K` (the `.T` attribute transposes a 2D tensor).
+++
**Answer:**

```python
return Q @ K.T
```
:::

---

:::terminal id="exercise-step3-output" title="Steps 1&ndash;3: Raw Scores" cmd="uv run python module_03_attention/src/main.py" maxw="960px" caption="Every raw score is within 0.007 of zero. The embeddings and projection weights are random numbers scaled by 0.1, so the queries and keys are short vectors and their dot products are tiny."
<span class="header">=== Step 1: make_token_vectors() (provided) ===</span> <span class="success">CORRECT</span>
  Tokens: the cat sat on mat
  X has shape (5, 8): one row of d_model=8 numbers per token
  the: [ 0.164 -0.016 -0.050  0.044 -0.076  0.108  0.080  0.168]

<span class="header t-orange">=== Step 2: compute_qkv() ===</span> <span class="success">CORRECT</span>
  Q, K, V shapes: (5, 4), (5, 4), (5, 4)
<span class="skipped">  ...</span>

<span class="header t-yellow">=== Step 3: raw_attention_scores() ===</span> <span class="success">CORRECT</span>
  Scores (5, 5), row = query, column = key:
            the     cat     sat      on     mat
   the   0.0010  0.0007  0.0005 -0.0001  0.0002
   cat   0.0018 -0.0067 -0.0031  0.0018  0.0009
   sat  -0.0029  0.0009  0.0007 -0.0016 -0.0015
    on  -0.0021  0.0053  0.0001  0.0001  0.0005
   mat  -0.0020  0.0033  0.0009 -0.0009 -0.0005
  <span class="success">CORRECT</span>    one row per query, one column per key: 3 queries and 2 keys give shape (3, 2)
  <span class="success">CORRECT</span>    entry (i, j) is q_i . k_j: Q=[[1,0],[0,2],[1,1]], K=[[1,1],[2,0]] gives [[1,2],[2,0],[2,2]]
  <span class="success">CORRECT</span>    matches torch.einsum('qd,kd->qk', Q, K) on random 5-token Q and K

<span class="header">=== Step 4: scaled_softmax() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: apply scaled softmax to attention scores</span>

<span class="skipped">...</span>
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
**Hint:** divide `scores` by the square root of `self.d_k`, then apply `F.softmax` along the last dimension.
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
**Hint:** matrix-multiply the weight matrix by the value matrix with the `@` operator.
+++
**Answer:**

```python
return weights @ V
```
:::

---

:::terminal id="exercise-step5-output" title="Steps 1&ndash;5: Unmasked Attention Output" cmd="uv run python module_03_attention/src/main.py" maxw="960px" caption="Tiny scores give nearly uniform weights, so every output is close to the average of the five values."
<span class="header">=== Step 1: make_token_vectors() (provided) ===</span> <span class="success">CORRECT</span>
<span class="header t-orange">=== Step 2: compute_qkv() ===</span> <span class="success">CORRECT</span>
<span class="header t-yellow">=== Step 3: raw_attention_scores() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header t-green">=== Step 4: scaled_softmax() ===</span> <span class="success">CORRECT</span>
  Weights (5, 5), row = query, column = key:
            the     cat     sat      on     mat
   the   0.2001  0.2000  0.2000  0.1999  0.2000
   cat   0.2003  0.1994  0.1998  0.2003  0.2002
   sat   0.1998  0.2002  0.2002  0.1999  0.1999
    on   0.1997  0.2005  0.1999  0.1999  0.2000
   mat   0.1998  0.2003  0.2001  0.1999  0.1999
  <span class="success">Row sums: [1.0, 1.0, 1.0, 1.0, 1.0]</span>
<span class="skipped">  ...</span>

<span class="header t-cyan">=== Step 5: attention_output() ===</span> <span class="success">CORRECT</span>
  Output (5, 4), one blended value vector per token:
   the  -0.0061  0.0025  0.0271  0.0188
   cat  -0.0061  0.0025  0.0271  0.0188
   sat  -0.0061  0.0025  0.0271  0.0187
    on  -0.0061  0.0025  0.0271  0.0187
   mat  -0.0061  0.0025  0.0271  0.0187
  <span class="success">CORRECT</span>    weights [[1,0],[0.5,0.5]] over values [[2,0],[0,4]] give [[2,0],[1,2]]
  <span class="success">CORRECT</span>    one output row per query: 3 queries over 2 values of width 4 give shape (3, 4)
  <span class="success">CORRECT</span>    with softmax weights, matches torch's F.scaled_dot_product_attention (5 tokens, d_k=4)

<span class="header">=== Step 6: causal_mask() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: create causal mask</span>

<span class="skipped">...</span>
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
return allowed.masked_fill(allowed == 0, float("-inf")).masked_fill(allowed == 1, 0.0)
```
:::

---

:::terminal id="exercise-step7-output" title="Steps 6&ndash;7: Causal Mask Output" cmd="uv run python module_03_attention/src/main.py" maxw="960px"
<span class="header">=== Step 1: make_token_vectors() (provided) ===</span> <span class="success">CORRECT</span>
<span class="header t-orange">=== Step 2: compute_qkv() ===</span> <span class="success">CORRECT</span>
<span class="header t-yellow">=== Step 3: raw_attention_scores() ===</span> <span class="success">CORRECT</span>
<span class="header t-green">=== Step 4: scaled_softmax() ===</span> <span class="success">CORRECT</span>
<span class="header t-cyan">=== Step 5: attention_output() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header t-blue">=== Step 6: causal_mask() ===</span> <span class="success">CORRECT</span>
  Mask for 5 tokens, row = query, column = key:
            the     cat     sat      on     mat
   the        0    -inf    -inf    -inf    -inf
   cat        0       0    -inf    -inf    -inf
   sat        0       0       0    -inf    -inf
    on        0       0       0       0    -inf
   mat        0       0       0       0       0
<span class="skipped">  ...</span>

<span class="header t-blue">=== Step 7: masked_attention() (Steps 2, 3, 5, 6 together) ===</span> <span class="success">CORRECT</span>
  Masked weights, row = query, column = key:
            the     cat     sat      on     mat
   the   1.0000  0.0000  0.0000  0.0000  0.0000
   cat   0.5011  0.4989  0.0000  0.0000  0.0000
   sat   0.3329  0.3335  0.3335  0.0000  0.0000
    on   0.2496  0.2506  0.2499  0.2499  0.0000
   mat   0.1998  0.2003  0.2001  0.1999  0.1999
  Saved attention comparison to module_03_attention/output/attention_comparison.png
  <span class="success">CORRECT</span>    the first token can only attend to itself: its weight row is [1, 0, 0, 0, 0]
  <span class="success">CORRECT</span>    every weight on a future token is exactly 0, and every row still sums to 1
  <span class="success">CORRECT</span>    the output matches torch's F.scaled_dot_product_attention(Q, K, V, is_causal=True)

<span class="header">=== Step 8: add_positional_embeddings() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: fill sine and cosine positional dimensions</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step8-code" title="Step 8: add_positional_embeddings()"
```python
    # Create the even dimension indices: 0, 2, 4, ...
    dim_pair = torch.arange(0, d_model, 2, dtype=X.dtype, device=X.device)
    # Compute the denominator term from the sinusoidal encoding equation.
    angle_rates = 1 / (10000 ** (dim_pair / d_model))
    # Broadcast positions against dimension frequencies to get all angles.
    angles = position * angle_rates
    # Start with zeros, then fill in the sinusoidal pattern.
    P = torch.zeros_like(X)

    # TODO: Fill the even dimensions of P with sine values and the odd dimensions with cosine values, using the angles above.
    raise NotImplementedError("TODO: fill sine and cosine positional dimensions")

    # Add position information to each token embedding.
    return X + P
```
+++
**Hint:** even columns (slice `0::2`) get the sine of `angles`; odd columns (slice `1::2`) get the cosine.
+++
**Answer:**

```python
P[:, 0::2] = torch.sin(angles)
P[:, 1::2] = torch.cos(angles)
```
:::

---

:::terminal id="exercise-step8-output" title="Step 8: Positional Embeddings Output" cmd="uv run python module_03_attention/src/main.py" maxw="960px" caption="Position 0 adds cos(0) = 1 to every odd dimension of &quot;the&quot;. With position added, the weights range from 0.193 to 0.207 instead of staying within 0.001 of 0.2."
<span class="header">=== Step 1: make_token_vectors() (provided) ===</span> <span class="success">CORRECT</span>
<span class="header t-orange">=== Step 2: compute_qkv() ===</span> <span class="success">CORRECT</span>
<span class="header t-yellow">=== Step 3: raw_attention_scores() ===</span> <span class="success">CORRECT</span>
<span class="header t-green">=== Step 4: scaled_softmax() ===</span> <span class="success">CORRECT</span>
<span class="header t-cyan">=== Step 5: attention_output() ===</span> <span class="success">CORRECT</span>
<span class="header t-blue">=== Step 6: causal_mask() ===</span> <span class="success">CORRECT</span>
<span class="header t-blue">=== Step 7: masked_attention() (Steps 2, 3, 5, 6 together) ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-red">=== Step 8: add_positional_embeddings() ===</span> <span class="success">CORRECT</span>
  the, before position: [ 0.164 -0.016 -0.050  0.044 -0.076  0.108  0.080  0.168]
  the, after position:  [ 0.164  0.984 -0.050  1.044 -0.076  1.108  0.080  1.168]
  Unmasked weights with position, row = query, column = key:
            the     cat     sat      on     mat
   the   0.1982  0.2001  0.2031  0.2017  0.1968
   cat   0.1992  0.1993  0.2006  0.2012  0.1998
   sat   0.1985  0.1979  0.2002  0.2026  0.2009
    on   0.1957  0.1969  0.2030  0.2061  0.1983
   mat   0.1939  0.1980  0.2073  0.2074  0.1934
  Saved positional effect to module_03_attention/output/positional_effect.png
  <span class="success">CORRECT</span>    position 0 is [sin 0, cos 0, sin 0, cos 0] = [0, 1, 0, 1]
  <span class="success">CORRECT</span>    position 1, d_model=4 is [sin 1, cos 1, sin 0.01, cos 0.01] = [0.841, 0.540, 0.010, 1.000]
  <span class="success">CORRECT</span>    matches sin and cos of pos / 10000^(2i/d_model) for 50 positions, d_model=16
  <span class="success">CORRECT</span>    adds the encoding on top of the tokens and leaves the input X unchanged

<span class="header">=== Extra Credit: kv_cache_step() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">Extra credit: project new_token through W_Q, W_K, and W_V</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: KV Cache

Implement `kv_cache_step()`: cache keys and values instead of recomputing them, and only project the new token. <!-- .element: class="text-lg" -->

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

The runner feeds the tokens in one at a time and plots the cost with and without the cache. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-extra-output" title="Extra Credit: Output" cmd="uv run python module_03_attention/src/main.py --step ec" maxw="1000px" caption="Each call projects only the new token and appends one key and one value, so the cache grows by one row per token. The last test feeds five tokens in one at a time and gets the same outputs as causal attention computed over the whole sequence."
<span class="header t-blue">=== Extra Credit: kv_cache_step() ===</span> <span class="success">CORRECT</span>
  Feeding the tokens in one at a time:
    Token 0 (the): cache size = 1, output norm = 0.0680
    Token 1 (cat): cache size = 2, output norm = 0.0420
    Token 2 (sat): cache size = 3, output norm = 0.0390
    Token 3 (on): cache size = 4, output norm = 0.0396
    Token 4 (mat): cache size = 5, output norm = 0.0336
  Saved KV cache plot to module_03_attention/output/kv_cache_growth.png
  <span class="success">CORRECT</span>    with an empty cache, the output is the token's own value: x=[1,2] gives v = x @ W_V = [3, 2]
  <span class="success">CORRECT</span>    appends one row to each cache: new key x @ W_K = [2, 1], new value x @ W_V = [3, 2]
  <span class="success">CORRECT</span>    5 tokens one at a time match full causal attention recomputed from scratch
:::
