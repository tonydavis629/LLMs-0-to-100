"""
Module 12 Exercise: Linear attention, two ways

You implement the seven functions below. Everything else (the softmax attention
baseline, the random test tensors, the timing sweep, the plotting, and the
runner) is provided. Each blank is one line or one short expression.

The claim you are going to prove to yourself: attention without the softmax can
be computed two completely different ways that give the same numbers. Steps 1-3
build the parallel form, which looks exactly like the attention you wrote in
Module 3 and costs O(n^2). Steps 4-5 build the recurrent form, which looks like
the RNNs the transformer replaced in Module 4 and costs O(n) with a fixed-size
state. Step 6 checks they agree. Step 7 times them, so the cost claim stops
being a claim.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_12_future/src/main.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Step 1: The feature map
# ---------------------------------------------------------------------------


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
    # HINT: torch.nn.functional is imported as F, and it has an elu function.
    #       Add 1 to its result.
    raise NotImplementedError("TODO: return the positive feature map of x")


# ---------------------------------------------------------------------------
# Step 2: The parallel form's scores
# ---------------------------------------------------------------------------


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
    # HINT: `@` does matrix multiplication, `.transpose(-2, -1)` swaps the last
    #       two dimensions, and `*` multiplies elementwise.
    raise NotImplementedError("TODO: return the causally masked score matrix")


# ---------------------------------------------------------------------------
# Step 3: The parallel form's output
# ---------------------------------------------------------------------------


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
    # HINT: `@` for the matrix multiply, `/` for the division. The shapes work
    #       out: (n, n) @ (n, d_v) is (n, d_v), divided by (n, 1).
    raise NotImplementedError("TODO: return the normalized attention output")


# ---------------------------------------------------------------------------
# Step 4: The recurrent state update
# ---------------------------------------------------------------------------


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
    # HINT: torch.outer(a, b) builds the outer product matrix. Return both
    #       values separated by a comma to make a tuple.
    raise NotImplementedError("TODO: return the updated (S, z) state")


# ---------------------------------------------------------------------------
# Step 5: The recurrent output
# ---------------------------------------------------------------------------


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
    # HINT: `@` handles both products here. q_t_phi @ S gives a vector, and
    #       q_t_phi @ z gives a single number to divide by.
    raise NotImplementedError("TODO: return this token's attention output")


# ---------------------------------------------------------------------------
# Step 6: The equivalence check
# ---------------------------------------------------------------------------


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
    # HINT: torch.allclose(a, b, atol=...) does exactly this.
    raise NotImplementedError("TODO: return whether a and b agree within tolerance")


# ---------------------------------------------------------------------------
# Step 7: The complexity measurement
# ---------------------------------------------------------------------------


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
        # HINT: call time.perf_counter() again and subtract `start`. That gives
        #       seconds, so multiply by 1000 for milliseconds.
        raise NotImplementedError("TODO: record this run's elapsed time")
    return min(timings)
