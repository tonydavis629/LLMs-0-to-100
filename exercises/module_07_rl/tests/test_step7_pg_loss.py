"""Step 7: pg_loss()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_pg_loss(pg_loss) -> list[Check]:
    """-advantage times the sum of the completion's log-probs (prompt masked out)."""
    checks = []
    mask = torch.tensor([False, True, True])

    # Only positions 1 and 2 count: -2 * (-2 + -3) = 10
    got = float(pg_loss(torch.tensor([-1.0, -2.0, -3.0]), 2.0, mask))
    name = "log-probs [-1, -2, -3], mask [F, T, T], advantage 2 give loss 10.0"
    if abs(got - 10.0) < 1e-5:
        checks.append(ok(name))
    else:
        # Point at the most likely slip
        if abs(got + 10.0) < 1e-5:
            nudge = "optimizers minimize, so negate: -advantage * sum"
        elif abs(got - 12.0) < 1e-5:
            nudge = "multiply by the mask so the prompt position is left out"
        else:
            nudge = "-advantage * (token_log_probs * mask).sum()"
        checks.append(bad(name, f"expected 10.0, got {got:.4f} ({nudge})"))

    # Descent on this loss must RAISE the log-probs of an above-average completion,
    # so the gradient with respect to each counted log-prob is -advantage = -2
    lp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    pg_loss(lp, 2.0, mask).backward()
    grad = lp.grad.tolist() if lp.grad is not None else None
    name = "descent raises a winner's log-probs: advantage 2 gives d loss / d log p = [0, -2, -2]"
    if grad is not None and torch.allclose(lp.grad, torch.tensor([0.0, -2.0, -2.0])):
        checks.append(ok(name))
    else:
        if grad is not None and grad[0] != 0.0:
            nudge = "the masked prompt position should get no gradient"
        else:
            nudge = "descent would push the winner's tokens down"
        checks.append(bad(name, f"expected [0.0, -2.0, -2.0], got {grad} ({nudge})"))

    # Changing a prompt-position log-prob must not change the loss
    a = float(pg_loss(torch.tensor([-1.0, -2.0, -3.0]), 2.0, mask))
    b = float(pg_loss(torch.tensor([-50.0, -2.0, -3.0]), 2.0, mask))
    name = "masked prompt positions contribute nothing to the loss"
    if abs(a - b) < 1e-5:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"loss changed from {a:.4f} to {b:.4f} (multiply the log-probs by the mask)"))
    return checks
