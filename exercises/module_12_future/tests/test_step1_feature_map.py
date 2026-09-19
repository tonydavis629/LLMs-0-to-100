"""Step 1: feature_map()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _show(t):
    """A tensor as a plain list, rounded to 4 decimals for printing."""
    return t.detach().double().round(decimals=4).tolist()


def check_feature_map(feature_map) -> list[Check]:
    """elu(x) + 1: equal to x + 1 for positive x, and always above zero."""
    checks = []

    # By hand: elu(-1) = e^-1 - 1, so elu(-1) + 1 = e^-1 = 0.3679.
    # For x >= 0, elu(x) = x, so the output is just x + 1.
    x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    expected = torch.tensor([[0.3679, 1.0], [2.0, 3.0]])
    out = feature_map(x)
    name = "[[-1, 0], [1, 2]] gives [[e^-1, 1], [2, 3]] = [[0.3679, 1], [2, 3]]"
    checks.append(
        ok(name)
        if out.shape == expected.shape and torch.allclose(out, expected, atol=1e-4)
        else bad(name, f"expected [[0.3679, 1.0], [2.0, 3.0]]\n"
                       f"got      {_show(out)}")
    )

    # Scores are dot products of these features, so any negative coordinate
    # could make an attention weight negative. Every output must be above zero.
    torch.manual_seed(0)
    x_big = torch.randn(100, 8) * 3
    out_big = feature_map(x_big)
    checks.append(
        ok("every output is strictly positive on 800 random inputs")
        if bool((out_big > 0).all())
        else bad("every output is strictly positive on 800 random inputs",
                 f"smallest output was {float(out_big.min()):.4f} (did you add 1 after the elu?)")
    )

    # A feature map works entry by entry, so the shape never changes
    checks.append(
        ok("keeps the input's shape: (100, 8) in, (100, 8) out")
        if tuple(out_big.shape) == (100, 8)
        else bad("keeps the input's shape: (100, 8) in, (100, 8) out",
                 f"got shape {tuple(out_big.shape)}")
    )
    return checks
