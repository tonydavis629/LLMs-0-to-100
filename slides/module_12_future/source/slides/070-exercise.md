:::divider id="divider-exercise" title="Exercise" sub="Linear attention, two ways"
Implement linear attention twice: as an $n \times n$ matrix computation, and as a recurrence with a fixed-size state. Verify the two agree, then measure how each scales.
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Build linear attention both ways, check they agree, and time them against softmax attention. <!-- .element: class="text-lg" -->

- Open `module_12_future/exercise.py`, fill in the seven `NotImplementedError` lines
- The softmax baseline, driver loop, random inputs, and plotting are provided
- Run after each step; unfinished steps are skipped automatically

```bash
# Linear attention, two ways
cd exercises
uv run python module_12_future/src/main.py
```

The parallel form runs after step 3, the recurrent form after step 5, the equivalence check after step 6, and the timing sweep after step 7. The runner saves a log-log cost plot to `output/attention_scaling.png`. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise Overview: No Training Involved

No dataset. Inputs are random tensors from a fixed seed: the measurement is about the **arithmetic**, not any text.

:::columns cols="2" gap="30px"
**Write the same function twice**

- Steps 1 to 3: the parallel form, an $n \times n$ score matrix minus the softmax
- Steps 4 and 5: the recurrent form, a fixed-size state carried forward like an RNN
+++
**Check and time it**

- Step 6: do the two agree? They do, to seven decimal places. That is the theorem
- Step 7: time both against softmax attention across sequence lengths
:::

---

:::step id="exercise-step1" title="Step 1: feature_map()"
```python
def feature_map(x: Tensor) -> Tensor:
    """Map queries and keys into a space where every coordinate is positive."""
    # TODO: Return elu(x) + 1, the positive feature map.
    raise NotImplementedError("TODO: return the positive feature map of x")
```
+++
**Hint:** `torch.nn.functional` is imported as `F`, and it has an `elu` function. Add 1 to its result.
+++
**Answer:**

```python
return F.elu(x) + 1
```

Softmax used `exp` to keep scores positive and to let one key dominate. This keeps them positive and gives up the domination. That lost sharpness is the price.
:::

---

:::step id="exercise-step2" title="Step 2: masked_scores()"
```python
def masked_scores(q_phi: Tensor, k_phi: Tensor) -> Tensor:
    """Score every query against every key, then delete the future."""
    n = q_phi.shape[-2]
    mask = torch.tril(torch.ones(n, n, dtype=q_phi.dtype, device=q_phi.device))
    # TODO: Return the masked scores: multiply q_phi by the transpose of k_phi
    #       to get every query-key pair, then multiply elementwise by `mask` to
    #       zero out the future.
    raise NotImplementedError("TODO: return the causally masked score matrix")
```
+++
**Hint:** `@` does matrix multiplication, `.transpose(-2, -1)` swaps the last two dimensions, and `*` multiplies elementwise.
+++
**Answer:**

```python
return (q_phi @ k_phi.transpose(-2, -1)) * mask
```

Note how the masking changed. Softmax attention sets future entries to $-\infty$ so `exp` sends them to zero. With no `exp`, you multiply by zero instead.
:::

---

:::step id="exercise-step3" title="Step 3: parallel_linear_attention()"
```python
def parallel_linear_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    """Compute linear attention the quadratic way, all positions at once."""
    q_phi = feature_map(Q)
    k_phi = feature_map(K)
    scores = masked_scores(q_phi, k_phi)
    normalizer = scores.sum(dim=-1, keepdim=True)
    # TODO: Return the attention output: matrix multiply `scores` with V to get
    #       the weighted sum of values, then divide by `normalizer`.
    raise NotImplementedError("TODO: return the normalized attention output")
```
+++
**Hint:** `@` for the matrix multiply, `/` for the division. The shapes work out: `(n, n) @ (n, d_v)` is `(n, d_v)`, divided by `(n, 1)`.
+++
**Answer:**

```python
return (scores @ V) / normalizer
```

Softmax normalized the weights for you. Here you do it by hand, or later positions would produce larger outputs purely because they attend to more of the past.
:::

---

:::terminal id="exercise-output-1" title="After Step 3: The Parallel Form Runs" cmd="uv run python module_12_future/src/main.py"
+=================================================================
|  Module 12: Linear attention, two ways
+=================================================================
  Steps implemented: 3 of 7
  Still to do: update_state, recurrent_step_output, outputs_match, time_forward

+-- Do the two forms agree? ---------------------------------------
  sequence length 256, head dimension 64

  The parallel form runs. The recurrent form needs steps 4-5.

+-- What does it cost? --------------------------------------------
  Skipped: the timing sweep needs step 7.
:::

---

:::step id="exercise-step4" title="Step 4: update_state()"
```python
def update_state(S: Tensor, z: Tensor, k_t_phi: Tensor, v_t: Tensor):
    """Absorb one token into the running state."""
    # TODO: Return the updated state as a tuple (new_S, new_z). Add the outer
    #       product of `k_t_phi` and `v_t` to S, and add `k_t_phi` to z.
    raise NotImplementedError("TODO: return the updated (S, z) state")
```
+++
**Hint:** `torch.outer(a, b)` builds the outer product matrix. Return both values separated by a comma to make a tuple.
+++
**Answer:**

```python
return S + torch.outer(k_t_phi, v_t), z + k_t_phi
```

This is the whole trick. `S` has a fixed size no matter how many tokens go by, so a million-token conversation costs the same memory as a ten-token one. It is also the catch: everything the model will ever know about the past has to fit in there.
:::

---

:::step id="exercise-step5" title="Step 5: recurrent_step_output()"
```python
def recurrent_step_output(q_t_phi: Tensor, S: Tensor, z: Tensor) -> Tensor:
    """Read one token's output out of the running state."""
    # TODO: Return this token's output: q_t_phi times S, divided by q_t_phi
    #       dotted with z.
    raise NotImplementedError("TODO: return this token's attention output")
```
+++
**Hint:** `@` handles both products here. `q_t_phi @ S` gives a vector, and `q_t_phi @ z` gives a single number to divide by.
+++
**Answer:**

```python
return (q_t_phi @ S) / (q_t_phi @ z)
```

Notice what is missing: no loop over past tokens, no $n \times n$ matrix, no dependence on sequence length at all. This is an RNN-style read, the pattern the transformer was designed to replace.
:::

---

:::terminal id="exercise-output-2" title="After Step 5: Both Forms Exist" cmd="uv run python module_12_future/src/main.py" caption="Two completely different algorithms, agreeing to seven decimal places. That is the theorem."
+=================================================================
|  Module 12: Linear attention, two ways
+=================================================================
  Steps implemented: 5 of 7
  Still to do: outputs_match, time_forward

+-- Do the two forms agree? ---------------------------------------
  sequence length 256, head dimension 64

  parallel vs recurrent    max difference = 3.58e-07   (step 6 not implemented)
  linear   vs softmax      max difference = 1.38e+00   DIFFERENT

  The first line is the theorem: the quadratic form and the RNN
  compute the same function, and differ only in floating-point noise.
  The second line is the caveat: linear attention is a DIFFERENT
  function from softmax attention, not an approximation of it.

+-- What does it cost? --------------------------------------------
  Skipped: the timing sweep needs step 7.
:::

---

:::step id="exercise-step6" title="Step 6: outputs_match()"
```python
def outputs_match(a: Tensor, b: Tensor, tolerance: float = 1e-4) -> bool:
    """Check whether two tensors agree everywhere, up to floating-point noise."""
    # TODO: Return True if a and b agree everywhere within `tolerance`.
    raise NotImplementedError("TODO: return whether a and b agree within tolerance")
```
+++
**Hint:** `torch.allclose(a, b, atol=...)` does exactly this.
+++
**Answer:**

```python
return torch.allclose(a, b, atol=tolerance)
```

Why a tolerance rather than `==`? The two forms are algebraically identical but add the same numbers in a different order, and floating-point addition is not associative. An exact check would fail for uninteresting reasons.
:::

---

:::step id="exercise-step7" title="Step 7: time_forward()"
```python
def time_forward(fn, repeats: int = 3) -> float:
    """Time one forward pass, in milliseconds."""
    # Warm-up run, not timed: pays one-time allocation costs up front.
    fn()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        # TODO: Append this run's elapsed time in milliseconds to `timings`.
        raise NotImplementedError("TODO: record this run's elapsed time")
    return min(timings)
```
+++
**Hint:** call `time.perf_counter()` again and subtract `start`. That gives seconds, so multiply by 1000 for milliseconds.
+++
**Answer:**

```python
timings.append((time.perf_counter() - start) * 1000)
```

Two details make this honest: the warm-up absorbs one-time allocation costs, and taking the **minimum** reports the cleanest run rather than the noisiest, since background processes can only slow a run down.
:::

---

:::terminal id="exercise-output-3" title="After Step 7: The Cost Curves" cmd="uv run python module_12_future/src/main.py" caption="Read the last column downward, then read the slopes."
+-- What does it cost? --------------------------------------------
              softmax       linear       linear
       n     parallel     parallel    recurrent
  +--------------------------------------------
     512      0.68 ms      0.69 ms      5.25 ms
    1024      2.02 ms      1.83 ms     10.62 ms
    2048      3.77 ms      3.82 ms     21.45 ms
    4096     18.55 ms     15.09 ms     42.42 ms
    8192    107.35 ms     91.38 ms     87.52 ms

+-- What are the exponents? ---------------------------------------
  Fitted slope of log(time) against log(n). This IS the exponent in
  the big-O: 2 means quadratic, 1 means linear.

  softmax attention (parallel)     slope = 1.78
  linear attention (parallel)      slope = 1.72
  linear attention (recurrent)     slope = 1.01
:::

---

<!-- .slide: id="exercise-figure" -->

## The Scaling Plot

<div class="img-figure">
  <img src="images/attention_scaling.png" alt="Log-log plot of time per forward pass against sequence length for softmax attention, parallel linear attention, and recurrent linear attention. The two parallel forms bend upward with slope near 2 while the recurrent form is a straight line of slope 1, and the lines cross near 8192 tokens.">
</div>

On log-log axes a power law is a straight line whose slope is the exponent. Two lines bend upward. One does not. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="exercise-takeaway" -->

## What the Crossover Means

The recurrent form is **ten times slower** at 512 tokens and **faster** at 8192. Both facts matter.

:::columns cols="2" gap="34px"
**Why it starts slow**

- A Python loop with huge per-step overhead
- Big-O says nothing about constants, and these constants are terrible
+++
**Why it wins anyway**

- It never builds the $n \times n$ matrix
- The exponent eventually beats the constant, at a context length people actually use
:::

The recurrent form gave up exact recall of earlier tokens. That tradeoff is why production models hybridize instead of choosing. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-extra" -->

## Extra Credit

<div class="card-grid cols-3">
<div class="card"><h4>Gated recurrence</h4><p>Multiply <code>S</code> by a decay factor before each update. You are now one scalar away from the forgetting in RWKV and Mamba.</p></div>
<div class="card"><h4>Memory accounting</h4><p>KV cache bytes at each length beside the state's constant size. More persuasive than the runtime plot.</p></div>
<div class="card"><h4>Sharpness</h4><p>Entropy of each row of softmax weights versus linear weights. This measures the blur directly, without training anything.</p></div>
<div class="card"><h4>Fit it yourself</h4><p>Do the log-log least squares by hand on the printed table. Same fit the scaling-law papers use on loss.</p></div>
<div class="card"><h4>One gradient step</h4><p>Rewrite the state update as gradient descent on a squared reconstruction loss. Confirm it gives the same <code>S</code>.</p></div>
<div class="card"><h4>Why that last one matters</h4><p>The state was never just memory. It was a tiny model being trained during the forward pass. That is test-time training.</p></div>
</div>
