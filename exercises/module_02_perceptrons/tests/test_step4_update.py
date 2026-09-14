"""Step 4: update_parameters()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_update_parameters(update_parameters) -> list[Check]:
    """One gradient-descent step: move against the gradient, scaled by the learning rate."""
    checks = []
    w, b = torch.tensor([1.0, 1.0]), torch.tensor(0.5)
    dw, db = torch.tensor([1.0, -1.0]), torch.tensor(0.2)

    # lr = 0.5: w -> [1 - 0.5, 1 + 0.5], b -> 0.5 - 0.1
    new_w, new_b = update_parameters(w, b, dw, db, 0.5)
    checks.append(
        ok("moves against the gradient: w=[1,1], dw=[1,-1], lr=0.5 gives [0.5, 1.5]")
        if torch.allclose(new_w, torch.tensor([0.5, 1.5]), atol=1e-6)
        else bad("moves against the gradient: w=[1,1], dw=[1,-1], lr=0.5 gives [0.5, 1.5]",
                 f"got {new_w.detach().tolist()} (a plus sign climbs the loss instead of descending it)")
    )
    checks.append(
        ok("updates the bias the same way: b=0.5, db=0.2, lr=0.5 gives 0.4")
        if abs(float(new_b) - 0.4) < 1e-6
        else bad("updates the bias the same way: b=0.5, db=0.2, lr=0.5 gives 0.4", f"got {float(new_b):.4f}")
    )

    # Ten times the learning rate is ten times the step
    small_w, _ = update_parameters(w, b, dw, db, 0.01)
    big_w, _ = update_parameters(w, b, dw, db, 0.1)
    checks.append(
        ok("the learning rate scales the step size")
        if torch.allclose(big_w - w, 10 * (small_w - w), atol=1e-6)
        else bad("the learning rate scales the step size",
                 f"lr=0.01 moved by {(small_w - w).tolist()}, lr=0.1 moved by {(big_w - w).tolist()}")
    )

    # Nothing to learn, nothing changes
    same_w, same_b = update_parameters(w, b, torch.zeros(2), torch.tensor(0.0), 0.5)
    checks.append(
        ok("a zero gradient leaves the parameters unchanged")
        if torch.equal(same_w, w) and float(same_b) == float(b)
        else bad("a zero gradient leaves the parameters unchanged",
                 f"got w={same_w.tolist()}, b={float(same_b):.4f}")
    )
    return checks
