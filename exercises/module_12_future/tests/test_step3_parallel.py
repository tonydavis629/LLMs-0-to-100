"""Step 3: parallel_linear_attention()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def _show(t):
    """A tensor as a plain list, rounded to 4 decimals for printing."""
    return t.detach().double().round(decimals=4).tolist()


def _feature_map(x):
    """A known-good Step 1, used while testing Step 3."""
    return F.elu(x) + 1


def _masked_scores(q_phi, k_phi):
    """A known-good Step 2, used while testing Step 3."""
    return torch.tril(q_phi @ k_phi.transpose(-2, -1))


def known_good_steps_1_and_2(parallel_linear_attention):
    """Swap in known-good Steps 1 and 2 for as long as a `with` block runs.

    parallel_linear_attention() calls your feature_map() and masked_scores()
    from Steps 1 and 2. Replacing them for a moment means only your Step 3
    line is being judged. The runner also uses this to check whether Step 3
    itself is written before it looks at Steps 1 and 2.
    """
    return patch.dict(parallel_linear_attention.__globals__,
                      {"feature_map": _feature_map, "masked_scores": _masked_scores})


def check_parallel_linear_attention(parallel_linear_attention) -> list[Check]:
    """Normalize each row of scores, then take the weighted average of the values."""
    with known_good_steps_1_and_2(parallel_linear_attention):
        return _checks(parallel_linear_attention)


def _checks(parallel_linear_attention) -> list[Check]:
    checks = []

    # By hand: with Q = K = 0 every feature is elu(0) + 1 = 1, so every
    # allowed score is equal. Token i then averages the values of tokens 0..i:
    # [1], then (1 + 2) / 2 = 1.5, then (1 + 2 + 6) / 3 = 3.
    zeros = torch.zeros(3, 2)
    V = torch.tensor([[1.0], [2.0], [6.0]])
    expected = torch.tensor([[1.0], [1.5], [3.0]])
    out = parallel_linear_attention(zeros, zeros, V)
    name = "equal scores give a running mean: Q=K=0, V=[1, 2, 6] gives [1, 1.5, 3]"
    checks.append(
        ok(name)
        if out.shape == expected.shape and torch.allclose(out, expected, atol=1e-5)
        else bad(name, f"expected {_show(expected.flatten())}\n"
                       f"got      {_show(out.flatten())} (divide scores @ V by normalizer)")
    )

    # Values narrower than the keys, so a mixed-up shape cannot slip through
    torch.manual_seed(0)
    Q, K, V = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 2)
    out = parallel_linear_attention(Q, K, V)
    checks.append(
        ok("returns one value-sized row per token: Q, K (5, 3) and V (5, 2) give (5, 2)")
        if tuple(out.shape) == (5, 2)
        else bad("returns one value-sized row per token: Q, K (5, 3) and V (5, 2) give (5, 2)",
                 f"got shape {tuple(out.shape)}")
    )

    # The same computation, one token at a time: token i's weights are its
    # scores against keys 0..i, scaled to sum to 1
    q_phi, k_phi = _feature_map(Q), _feature_map(K)
    rows = []
    for i in range(5):
        weights = k_phi[: i + 1] @ q_phi[i]
        rows.append((weights / weights.sum()) @ V[: i + 1])
    reference = torch.stack(rows)
    checks.append(
        ok("matches a token-by-token weighted average on a random 5-token input")
        if out.shape == reference.shape and torch.allclose(out, reference, atol=1e-5)
        else bad("matches a token-by-token weighted average on a random 5-token input",
                 f"expected row 4 = {_show(reference[4])}\n"
                 f"got      row 4 = {_show(out[-1].flatten())}")
    )
    return checks
