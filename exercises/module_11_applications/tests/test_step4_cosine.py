"""Step 4: cosine_similarity()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tests.check import Check, bad, ok


def _close_check(name: str, got, expected: float, nudge: str = "") -> Check:
    """Pass if got is within 1e-6 of expected, otherwise show both values."""
    if math.isclose(float(got), expected, abs_tol=1e-6):
        return ok(name)
    return bad(name, f"expected {expected:.4f}, got {float(got):.4f}{nudge}")


def check_cosine_similarity(cosine_similarity) -> list[Check]:
    """Vectors whose angle is known, then a cross-check against PyTorch."""
    checks = []

    # [2, 4, 6] is [1, 2, 3] stretched: same direction, so cosine 1 whatever the length
    got = cosine_similarity(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    nudge = " (divide the dot product by `denominator`, both lengths multiplied)" if float(got) > 1.0 else ""
    checks.append(_close_check(
        "parallel vectors score 1.0 whatever their lengths: [1, 2, 3] and [2, 4, 6]",
        got, 1.0, nudge,
    ))

    # At right angles the dot product is 0: no shared direction at all
    checks.append(_close_check(
        "perpendicular vectors score 0.0: [1, 0] and [0, 3]",
        cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 3.0])), 0.0,
    ))

    # dot = 1*3 + 2*4 = 11, lengths sqrt(5) and 5, so 11 / (5 * sqrt(5))
    checks.append(_close_check(
        "hand-computed: [1, 2] and [3, 4] give 11 / (sqrt(5) x 5) = 0.9839",
        cosine_similarity(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 11 / (5 * math.sqrt(5)),
    ))

    # Cross-check on random 384-dimensional vectors, the size of the dense embeddings
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=384), rng.normal(size=384)
    expected = float(torch.nn.functional.cosine_similarity(
        torch.from_numpy(a), torch.from_numpy(b), dim=0))
    checks.append(_close_check(
        "agrees with torch.nn.functional.cosine_similarity on random 384-dim vectors",
        cosine_similarity(a, b), expected,
    ))
    return checks
