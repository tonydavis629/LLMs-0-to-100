"""Step 8: kl_penalty()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_kl_penalty(kl_penalty) -> list[Check]:
    """Sum of (log pi_policy - log pi_ref) over the completion tokens."""
    checks = []
    policy = torch.tensor([-1.0, -1.0, -0.5])
    ref = torch.tensor([-2.0, -1.5, -1.0])
    mask = torch.tensor([False, True, True])

    # Positions 1 and 2 count: (-1 - -1.5) + (-0.5 - -1) = 0.5 + 0.5 = 1.0
    got = float(kl_penalty(policy, ref, mask))
    name = "policy [-1, -1, -0.5], reference [-2, -1.5, -1], mask [F, T, T] give 0.5 + 0.5 = 1.0"
    if abs(got - 1.0) < 1e-5:
        checks.append(ok(name))
    else:
        # Point at the most likely slip
        if abs(got + 1.0) < 1e-5:
            nudge = "subtract in the other order: policy minus reference"
        elif abs(got - 2.0) < 1e-5:
            nudge = "multiply by the mask so the prompt position is left out"
        else:
            nudge = "((policy_log_probs - ref_log_probs) * mask).sum()"
        checks.append(bad(name, f"expected 1.0, got {got:.4f} ({nudge})"))

    # Before any training the policy IS the reference, so there is no drift to penalize
    torch.manual_seed(0)
    same = torch.randn(6)
    got = float(kl_penalty(same, same.clone(), torch.ones(6, dtype=torch.bool)))
    name = "zero when the policy equals the reference (as it does before training)"
    if abs(got) < 1e-6:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected 0.0, got {got:.4f}"))

    # Changing a prompt-position log-prob must not change the penalty
    moved = torch.tensor([-50.0, -1.0, -0.5])
    a, b = float(kl_penalty(policy, ref, mask)), float(kl_penalty(moved, ref, mask))
    name = "masked prompt positions contribute nothing to the penalty"
    if abs(a - b) < 1e-5:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"penalty changed from {a:.4f} to {b:.4f} (multiply the difference by the mask)"))
    return checks
