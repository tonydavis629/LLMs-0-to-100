"""Step 3: compute_gradients()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from src.activations import sigmoid
from tests.check import Check, bad, ok


def check_compute_gradients(compute_gradients) -> list[Check]:
    """The hand-derived sigmoid + BCE gradient, checked against autograd."""
    checks = []

    # Hand example: errors are (+/-0.25), so dw = X.T @ err / 2 and db = mean(err) = 0
    X = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    y = torch.tensor([1.0, 0.0])
    p = torch.tensor([0.75, 0.25])
    dw, db = compute_gradients(X, y, p)
    exp_dw, exp_db = torch.tensor([-0.25, -0.1875]), 0.0
    checks.append(
        ok("matches the hand-computed dw = X.T @ (p - y) / n on a 2-sample batch")
        if tuple(dw.shape) == (2,) and torch.allclose(dw, exp_dw, atol=1e-4)
        else bad("matches the hand-computed dw = X.T @ (p - y) / n on a 2-sample batch",
                 f"expected {exp_dw.tolist()}, got {dw.detach().flatten().tolist()}")
    )
    checks.append(
        ok("db is the mean error (here the two errors cancel to 0)")
        if abs(float(db) - exp_db) < 1e-4
        else bad("db is the mean error (here the two errors cancel to 0)", f"got {float(db):.4f}")
    )

    # Perfect predictions mean nothing to learn: both gradients are zero
    dw0, db0 = compute_gradients(X, y, y.clone())
    checks.append(
        ok("gradients are zero when predictions equal the labels")
        if torch.allclose(dw0, torch.zeros(2), atol=1e-6) and abs(float(db0)) < 1e-6
        else bad("gradients are zero when predictions equal the labels",
                 f"got dw={dw0.detach().tolist()}, db={float(db0):.4f}")
    )

    # The real test: agree with PyTorch's autograd on a random batch
    torch.manual_seed(0)
    Xr, yr = torch.randn(20, 2), (torch.rand(20) > 0.5).float()
    w = torch.randn(2, requires_grad=True)
    b = torch.randn((), requires_grad=True)
    pr = sigmoid(Xr @ w + b)
    loss = -(yr * torch.log(pr) + (1 - yr) * torch.log(1 - pr)).mean()
    loss.backward()
    dw_s, db_s = compute_gradients(Xr, yr, pr.detach())
    checks.append(
        ok("agrees with torch autograd on a random 20-sample batch")
        if torch.allclose(dw_s, w.grad, atol=1e-4) and abs(float(db_s) - float(b.grad)) < 1e-4
        else bad("agrees with torch autograd on a random 20-sample batch",
                 f"autograd dw={w.grad.tolist()}, db={float(b.grad):.4f}; "
                 f"yours dw={dw_s.detach().flatten().tolist()}, db={float(db_s):.4f} "
                 "(did you divide by the batch size?)")
    )
    return checks
