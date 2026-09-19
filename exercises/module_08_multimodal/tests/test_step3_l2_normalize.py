"""Step 3: l2_normalize()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch

from tests.check import Check, bad, ok


def _rounded(t: torch.Tensor) -> list:
    """A tensor as nested lists with 4 decimals, for readable failure messages."""
    return torch.round(t.detach() * 10000).div(10000).tolist()


def check_l2_normalize(l2_normalize) -> list[Check]:
    """Scale every embedding (row) to length 1 without changing its direction."""
    checks = []

    # Row [3, 4] has length 5 and row [0, 2] has length 2
    got = l2_normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    name = "divides each row by its own length: [[3,4],[0,2]] gives [[0.6, 0.8], [0, 1]]"
    checks.append(
        ok(name)
        if got.shape == expected.shape and torch.allclose(got, expected, atol=1e-6)
        else bad(name, f"expected [[0.6, 0.8], [0.0, 1.0]], got {_rounded(got)} "
                       "(normalize along the last dimension, dim=-1)")
    )

    # Every row of a random batch should come out with length exactly 1
    torch.manual_seed(0)
    got = l2_normalize(torch.randn(4, 8) * 5)
    norms = got.norm(dim=-1)
    name = "every row of a random (4, 8) batch ends up with length 1"
    checks.append(
        ok(name)
        if tuple(got.shape) == (4, 8) and torch.allclose(norms, torch.ones(4), atol=1e-5)
        else bad(name, f"got shape {tuple(got.shape)} and row lengths {[round(v, 4) for v in norms.tolist()]}")
    )

    # On the unit sphere a plain dot product IS the cosine similarity:
    # [2, 0] and [3, 3] are 45 degrees apart, and cos(45) = 0.7071
    a, b = l2_normalize(torch.tensor([[2.0, 0.0], [3.0, 3.0]]))
    dot = float(a @ b)
    name = "then a dot product is a cosine: [2,0] and [3,3] give cos(45) = 0.7071"
    checks.append(
        ok(name)
        if math.isclose(dot, math.cos(math.pi / 4), abs_tol=1e-5)
        else bad(name, f"expected 0.7071, got {dot:.4f}")
    )
    return checks
