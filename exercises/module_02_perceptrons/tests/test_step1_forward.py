"""Step 1: forward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_forward(forward) -> list[Check]:
    """A single neuron: sigmoid of a weighted sum, for a whole batch at once."""
    checks = []

    # A 2-sample batch with hand-picked weights so the answer is easy to verify
    X = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    w = torch.tensor([1.0, -1.0])
    b = torch.tensor(0.5)
    out = forward(X, w, b)

    # One number per sample, not a matrix
    checks.append(
        ok("returns one output per sample, shape (n_samples,)")
        if tuple(out.shape) == (2,)
        else bad("returns one output per sample, shape (n_samples,)",
                 f"expected shape (2,), got {tuple(out.shape)}")
    )

    # The two samples give z = 1*1 + 2*(-1) + 0.5 = -0.5 and z = -1 - 0.5 + 0.5 = -1.0
    expected = torch.tensor([0.3775, 0.2689])
    checks.append(
        ok("matches sigmoid(X @ w + b) on a hand-computed batch")
        if out.shape == expected.shape and torch.allclose(out, expected, atol=1e-3)
        else bad("matches sigmoid(X @ w + b) on a hand-computed batch",
                 f"expected {expected.tolist()}, got {out.detach().flatten().tolist()}")
    )

    # Whatever the inputs, the output must be a probability
    torch.manual_seed(0)
    big = forward(torch.randn(50, 2) * 3, torch.randn(2), torch.randn(()))
    checks.append(
        ok("every output is a probability between 0 and 1")
        if bool(((big >= 0) & (big <= 1)).all())
        else bad("every output is a probability between 0 and 1",
                 f"min {float(big.min()):.4f}, max {float(big.max()):.4f} (did you apply sigmoid?)")
    )
    return checks
