:::divider id="divider-exercise" title="Exercise" sub="Linear attention, two ways"
Implement linear attention twice: as an $n \times n$ matrix computation, and as a recurrence with a fixed-size state. Verify the two agree, then measure how each scales.
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Build linear attention both ways, check they agree, and time them against softmax attention. <!-- .element: class="text-lg" -->

- Open `module_12_future/exercise.py`, fill in the seven `NotImplementedError` lines
- The softmax baseline, driver loop, random inputs, and plotting are provided
- Run after each step to see which steps pass their tests

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_12_future/src/main.py

# Run a single step (1-7)
uv run python module_12_future/src/main.py --step 5
```

Step 3 compares the parallel form with softmax attention, step 5 compares the recurrent form with the parallel form, and step 7 times all three and saves a log-log cost plot to `output/attention_scaling.png`. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code computed, then runs the step's **tests** from `tests/`. Each test calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed, and the expected and actual values are printed beneath it
- **INCOMPLETE**: the function still raises `NotImplementedError`, or the step's demo needs an earlier step you have not written yet

The tag sits on the step's header line. The tests check only that step's own line, so a bug in Step 2 cannot fail the tests for Step 5. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_12_future/src/main.py --step 6" maxw="920px" caption="Here <code>outputs_match()</code> called <code>torch.allclose(a, b)</code> without <code>atol=tolerance</code>. The demo reports a MISMATCH that looks like a problem with the two forms. The failing tests point at the tolerance instead."
<span class="header">=== Step 6: outputs_match() ===</span> <span class="t-fail">INCORRECT</span>
  parallel vs recurrent    max difference = 3.58e-07   <span class="t-fail">MISMATCH</span>
  linear   vs softmax      max difference = 1.38e+00   DIFFERENT

  The first line is the theorem: the quadratic form and the RNN
  compute the same function, and differ only in floating-point noise.
  The second line is the caveat: linear attention is a DIFFERENT
  function from softmax attention, not an approximation of it.
  <span class="success">CORRECT</span>    a tensor matches itself
  <span class="t-fail">INCORRECT</span>  a gap below the tolerance still matches: [0, 0] vs [0, 0.00005] gives True
             expected True, got False (did you pass atol=tolerance?)
  <span class="success">CORRECT</span>    a gap above the tolerance does not: [0, 0] vs [0, 0.01] gives False
  <span class="t-fail">INCORRECT</span>  uses the tolerance it is given: a 0.05 gap matches when tolerance=0.1
             expected True, got False (did you use the tolerance argument?)
:::

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

- Step 6: do the two agree? They do, to within 3.6e-07. That is the theorem
- Step 7: time both against softmax attention across sequence lengths
:::

Each blank is one line, and every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::step id="exercise-step1" title="Step 1: feature_map()"
```python
def feature_map(x: Tensor) -> Tensor:
    """Map queries and keys into a space where every coordinate is positive.

    Softmax attention scores a query against a key with exp(q . k), and the
    exponential guarantees two things: scores are never negative, and one large
    score can dominate all the others. Linear attention throws the exponential
    away, so it needs some other guarantee that scores stay positive. Without
    it, a "weighted average" could have negative weights summing to near zero,
    and the output would explode.

    The standard choice is elu(x) + 1, which is smooth, always positive, and
    close to x + 1 for positive inputs. Note what it does NOT do: it cannot make
    one score dominate the way exp can. That lost sharpness is the price linear
    attention pays, and section b of the lecture is about what it buys back.

    Args:
        x: A tensor of queries or keys, shape (n, d).

    Returns:
        A tensor of the same shape, every entry strictly positive.
    """
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
    """Score every query against every key, then delete the future.

    This is the O(n^2) step: an n-by-n matrix where entry (i, j) says how much
    query i cares about key j. It is the same shape as the score matrix in
    Module 3, with one difference that matters enormously later: there is no
    softmax. The entries are raw dot products of positive vectors, so they are
    already positive, and nothing has normalized them yet.

    Causal masking works differently without a softmax. In Module 3 you set
    future entries to -inf so that exp() would send them to zero. Here there is
    no exp, so you delete them by multiplying by zero instead.

    Args:
        q_phi: Feature-mapped queries, shape (n, d).
        k_phi: Feature-mapped keys, shape (n, d).

    Returns:
        The causally masked score matrix, shape (n, n), with zeros above the
        diagonal.
    """
    # The causal mask: entry (i, j) is 1.0 when j <= i (past and present) and
    # 0.0 when j > i (the future). torch.tril keeps the lower triangle.
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
    """Compute linear attention the quadratic way, all positions at once.

    With the scores in hand this is the same weighted average of values that
    Module 3 computed. The only wrinkle is normalization: softmax normalized
    the weights for you, and here you have to do it by hand. Each row of the
    score matrix must sum to 1 before it multiplies the values, or long rows
    (later positions, which attend to more of the past) would produce larger
    outputs than short ones purely because they have more terms.

    Args:
        Q: Queries, shape (n, d).
        K: Keys, shape (n, d).
        V: Values, shape (n, d_v).

    Returns:
        The attention output, shape (n, d_v).
    """
    # Feature-map the queries and keys (step 1), then score them (step 2).
    q_phi = feature_map(Q)
    k_phi = feature_map(K)
    scores = masked_scores(q_phi, k_phi)
    # Each row's total weight. keepdim=True leaves it shaped (n, 1) so that it
    # broadcasts cleanly across the value dimension when we divide.
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

:::terminal id="exercise-output-1" title="After Step 3: The Parallel Form Runs" cmd="uv run python module_12_future/src/main.py" caption="The masked scores are zero above the diagonal. The parallel form runs, and its output differs from softmax attention's by as much as 1.38, because linear attention is a different function."
<span class="header">=== Step 1: feature_map() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 2: masked_scores() ===</span> <span class="success">CORRECT</span>
  4 tokens, q_phi = k_phi = [[1,1],[1,2],[2,1],[2,2]]
  row i is query i, column j is key j:
       2.0   0.0   0.0   0.0
       3.0   5.0   0.0   0.0
       3.0   4.0   5.0   0.0
       4.0   6.0   6.0   8.0
<span class="skipped">  ...</span>
<span class="header">=== Step 3: parallel_linear_attention() ===</span> <span class="success">CORRECT</span>
  parallel form: 256 tokens, head dimension 64, output shape (256, 64)
  linear vs softmax        max difference = 1.38e+00
  <span class="success">CORRECT</span>    equal scores give a running mean: Q=K=0, V=[1, 2, 6] gives [1, 1.5, 3]
  <span class="success">CORRECT</span>    returns one value-sized row per token: Q, K (5, 3) and V (5, 2) give (5, 2)
  <span class="success">CORRECT</span>    matches a token-by-token weighted average on a random 5-token input

<span class="header">=== Step 4: update_state() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return the updated (S, z) state</span>

<span class="header">=== Step 5: recurrent_step_output() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return this token's attention output</span>
<span class="skipped">  ...</span>
:::

---

:::step id="exercise-step4" title="Step 4: update_state()"
```python
def update_state(S: Tensor, z: Tensor, k_t_phi: Tensor, v_t: Tensor):
    """Absorb one token into the running state.

    This is the whole trick. Instead of storing every past key and value the
    way a KV cache does (Module 10), keep one matrix S that accumulates the
    outer products of keys and values, and one vector z that accumulates the
    keys. Both have a fixed size that does not depend on how many tokens have
    gone by, so a million-token conversation costs exactly as much memory as a
    ten-token one.

    That is also the catch, and it is worth sitting with: everything the model
    will ever know about the past has to fit in S. A KV cache can reproduce any
    earlier token exactly. This cannot, and that is why the recurrent models in
    the lecture lag on exact-recall tasks.

    Args:
        S: The running state matrix, shape (d, d_v).
        z: The running normalizer, shape (d,).
        k_t_phi: This token's feature-mapped key, shape (d,).
        v_t: This token's value, shape (d_v,).

    Returns:
        The updated (S, z) pair, same shapes as the inputs.
    """
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
    """Read one token's output out of the running state.

    Reading is a single matrix-vector product against the state, with the same
    normalization step 3 did by hand. The remarkable part is what is missing:
    no loop over past tokens, no n-by-n matrix, no dependence on sequence
    length at all. This is exactly the RNN read the transformer was supposed to
    have made obsolete, and if step 6 passes it computes the same function as
    the quadratic form above.

    Args:
        q_t_phi: This token's feature-mapped query, shape (d,).
        S: The running state matrix, shape (d, d_v).
        z: The running normalizer, shape (d,).

    Returns:
        This token's output vector, shape (d_v,).
    """
    # TODO: Return this token's output: q_t_phi times S, divided by q_t_phi
    #       dotted with z.
    raise NotImplementedError("TODO: return this token's attention output")
```
+++
**Hint:** `@` handles both products here. The query against `S` gives a vector, and the query against `z` gives a single number to divide by.
+++
**Answer:**

```python
return (q_t_phi @ S) / (q_t_phi @ z)
```

Notice what is missing: no loop over past tokens, no $n \times n$ matrix, no dependence on sequence length at all. This is an RNN-style read, the pattern the transformer was designed to replace.
:::

---

:::terminal id="exercise-output-2" title="After Step 5: Both Forms Exist" cmd="uv run python module_12_future/src/main.py" caption="Two different algorithms, run on the same 256 tokens, land within 3.58e-07 of each other. Step 6 decides whether that counts as a match."
<span class="header">=== Step 1: feature_map() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: masked_scores() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: parallel_linear_attention() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 4: update_state() ===</span> <span class="success">CORRECT</span>
  <span class="success">CORRECT</span>    first token from an empty state: k=[1,2], v=[3,4] gives S=[[3,4],[6,8]], z=[1,2]
  <span class="success">CORRECT</span>    adds to the state rather than replacing it: S and z start as all ones
  <span class="success">CORRECT</span>    10 tokens one at a time give S = K^T V and z = the sum of the keys

<span class="header">=== Step 5: recurrent_step_output() ===</span> <span class="success">CORRECT</span>
  recurrent form: 256 tokens one at a time, output shape (256, 64)
  parallel vs recurrent    max difference = 3.58e-07
  <span class="success">CORRECT</span>    q=[1,2], S=[[1,0,2],[3,1,0]], z=[2,1] gives [7,2,2] / 4 = [1.75, 0.5, 0.5]
  <span class="success">CORRECT</span>    after one token the output is that token's value: v=[3, -1, 4] comes back out
  <span class="success">CORRECT</span>    matches the parallel form row by row on a random 6-token sequence

<span class="header">=== Step 6: outputs_match() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return whether a and b agree within tolerance</span>

<span class="header">=== Step 7: time_forward() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: record this run's elapsed time</span>
:::

---

:::step id="exercise-step6" title="Step 6: outputs_match()"
```python
def outputs_match(a: Tensor, b: Tensor, tolerance: float = 1e-4) -> bool:
    """Check whether two tensors agree everywhere, up to floating-point noise.

    The two forms are algebraically identical but they do the arithmetic in a
    different order, and floating-point addition is not associative, so the
    results will differ in the last few decimal places. An exact `==` would
    fail for uninteresting reasons. A tolerance check asks the question you
    actually care about: are these the same function?

    Args:
        a: One tensor.
        b: Another tensor of the same shape.
        tolerance: How far apart two entries may be and still count as equal.

    Returns:
        True if every pair of entries is within `tolerance`.
    """
    # TODO: Return True if a and b agree everywhere within `tolerance`.
    raise NotImplementedError("TODO: return whether a and b agree within tolerance")
```
+++
**Hint:** `torch.allclose` compares two tensors elementwise; its `atol` argument sets the tolerance.
+++
**Answer:**

```python
return torch.allclose(a, b, atol=tolerance)
```

Why a tolerance rather than `==`? The two forms are algebraically identical but add the same numbers in a different order, and floating-point addition is not associative. An exact check would fail for uninteresting reasons.
:::

---

:::terminal id="exercise-output-verdict" title="After Step 6: The Verdict" cmd="uv run python module_12_future/src/main.py" caption="Two completely different algorithms, agreeing to within 3.6e-07. That is the theorem."
<span class="header">=== Step 1: feature_map() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: masked_scores() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: parallel_linear_attention() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: update_state() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: recurrent_step_output() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 6: outputs_match() ===</span> <span class="success">CORRECT</span>
  parallel vs recurrent    max difference = 3.58e-07   <span class="success">MATCH</span>
  linear   vs softmax      max difference = 1.38e+00   DIFFERENT

  The first line is the theorem: the quadratic form and the RNN
  compute the same function, and differ only in floating-point noise.
  The second line is the caveat: linear attention is a DIFFERENT
  function from softmax attention, not an approximation of it.
  <span class="success">CORRECT</span>    a tensor matches itself
  <span class="success">CORRECT</span>    a gap below the tolerance still matches: [0, 0] vs [0, 0.00005] gives True
  <span class="success">CORRECT</span>    a gap above the tolerance does not: [0, 0] vs [0, 0.01] gives False
  <span class="success">CORRECT</span>    uses the tolerance it is given: a 0.05 gap matches when tolerance=0.1

<span class="header">=== Step 7: time_forward() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: record this run's elapsed time</span>
:::

---

:::step id="exercise-step7" title="Step 7: time_forward()"
```python
def time_forward(fn, repeats: int = 3) -> float:
    """Time one forward pass, in milliseconds.

    Complexity claims are theory until you measure them. The runner calls this
    on each attention implementation at a range of sequence lengths, and the
    resulting curves are the point of the whole exercise: one bends upward on a
    log-log plot and one does not.

    Two details make the measurement honest. The warm-up call absorbs one-time
    costs like memory allocation, and taking the minimum of several runs rather
    than the mean reports the cleanest run instead of the noisiest, since
    background processes can only ever slow a run down.

    Args:
        fn: A zero-argument callable that runs one forward pass.
        repeats: How many timed runs to take.

    Returns:
        The fastest run, in milliseconds.
    """
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
<span class="header">=== Step 1: feature_map() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: masked_scores() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: parallel_linear_attention() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: update_state() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: recurrent_step_output() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: outputs_match() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 7: time_forward() ===</span> <span class="success">CORRECT</span>
              softmax       linear       linear
       n     parallel     parallel    recurrent
  +--------------------------------------------
     512      1.01 ms      0.83 ms      7.01 ms
    1024      3.09 ms      2.65 ms     13.94 ms
    2048      5.97 ms      5.51 ms     29.17 ms
    4096     27.82 ms     21.52 ms     53.68 ms
    8192    141.37 ms    128.66 ms    106.28 ms
  <span class="skipped">...</span>
  softmax attention (parallel)     slope = 1.74
  linear attention (parallel)      slope = 1.76
  linear attention (recurrent)     slope = 0.98
  <span class="skipped">...</span>
  Saved figure to output/attention_scaling.png
  <span class="success">CORRECT</span>    reports milliseconds: a 20 ms sleep measures at least 20
  <span class="success">CORRECT</span>    an empty function measures a small positive time (0 to 5 ms)
  <span class="success">CORRECT</span>    calls fn once to warm up and once per repeat: repeats=3 makes 4 calls
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

The recurrent form is **eight times slower** than the parallel linear form at 512 tokens and **faster** at 8192. Both facts matter.

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
