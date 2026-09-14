"""Step 2: binary_cross_entropy()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch

from tests.check import Check, bad, ok


def _close(name: str, got: torch.Tensor, expected: float, note: str = "") -> Check:
    """Pass if a scalar tensor is within 1e-3 of the expected number."""
    value = float(got)
    if math.isclose(value, expected, abs_tol=1e-3):
        return ok(name)
    return bad(name, f"expected {expected:.4f}, got {value:.4f}" + (f" ({note})" if note else ""))


def check_binary_cross_entropy(binary_cross_entropy) -> list[Check]:
    """Shannon's surprise as a loss: cheap when right, expensive when wrong."""
    checks = []
    one, zero = torch.tensor(1.0), torch.tensor(0.0)

    # -ln(0.9): a confident, correct prediction costs little
    checks.append(_close("a confident correct prediction (y=1, p=0.9) costs -ln 0.9 = 0.105 nats",
                         binary_cross_entropy(one, torch.tensor(0.9)), -math.log(0.9)))

    # -ln(0.5) = ln 2 whichever label is true
    half = torch.tensor(0.5)
    both = torch.stack([binary_cross_entropy(one, half), binary_cross_entropy(zero, half)])
    checks.append(
        ok("a 50/50 prediction costs ln 2 = 0.693 for either label")
        if torch.allclose(both, torch.full((2,), math.log(2)), atol=1e-3)
        else bad("a 50/50 prediction costs ln 2 = 0.693 for either label",
                 f"got {both.tolist()} for y=1 and y=0")
    )

    # -ln(0.1): the same wrong-way confidence costs 22x more
    checks.append(_close("a confident wrong prediction (y=1, p=0.1) costs -ln 0.1 = 2.303 nats",
                         binary_cross_entropy(one, torch.tensor(0.1)), -math.log(0.1),
                         "check the (1 - y) log(1 - p) term"))

    # Works elementwise on a batch, so .mean() later gives the average loss
    y = torch.tensor([1.0, 0.0, 1.0])
    p = torch.tensor([0.9, 0.2, 0.6])
    out = binary_cross_entropy(y, p)
    expected = torch.tensor([-math.log(0.9), -math.log(0.8), -math.log(0.6)])
    checks.append(
        ok("works elementwise on a batch (one loss per sample)")
        if out.shape == expected.shape and torch.allclose(out, expected, atol=1e-3)
        else bad("works elementwise on a batch (one loss per sample)",
                 f"expected {[round(v, 4) for v in expected.tolist()]}, got {out.detach().tolist()}")
    )
    return checks
