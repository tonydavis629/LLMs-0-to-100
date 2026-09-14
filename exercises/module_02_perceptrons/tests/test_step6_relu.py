"""Step 6: relu()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_relu(relu) -> list[Check]:
    """max(0, z): negatives to zero, positives untouched, shape preserved."""
    checks = []

    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = relu(z)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0])
    checks.append(
        ok("negatives become 0 and positives pass through: [-2,-0.5,0,0.5,2] -> [0,0,0,0.5,2]")
        if out.shape == expected.shape and torch.equal(out, expected)
        else bad("negatives become 0 and positives pass through: [-2,-0.5,0,0.5,2] -> [0,0,0,0.5,2]",
                 f"got {out.detach().tolist()}")
    )

    # Hidden layers are matrices, so it has to work elementwise on 2D input
    torch.manual_seed(0)
    m = torch.randn(4, 3)
    out2 = relu(m)
    checks.append(
        ok("works elementwise on a matrix and keeps its shape")
        if out2.shape == m.shape and torch.equal(out2, torch.clamp(m, min=0))
        else bad("works elementwise on a matrix and keeps its shape",
                 f"expected shape {tuple(m.shape)}, got {tuple(out2.shape)}")
    )

    # Backprop through ReLU: gradient 1 where z > 0, 0 where z < 0
    zg = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    relu(zg).sum().backward()
    checks.append(
        ok("lets gradients through where z > 0 and blocks them where z < 0")
        if zg.grad is not None and torch.equal(zg.grad, torch.tensor([0.0, 1.0, 0.0, 1.0]))
        else bad("lets gradients through where z > 0 and blocks them where z < 0",
                 f"expected gradient [0, 1, 0, 1], got {None if zg.grad is None else zg.grad.tolist()}")
    )
    return checks
