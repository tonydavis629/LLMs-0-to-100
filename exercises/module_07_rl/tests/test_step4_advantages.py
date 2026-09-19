"""Step 4: group_relative_advantages()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _fmt(t: torch.Tensor) -> str:
    """Show a tensor as a short list with 3 decimals."""
    return "[" + ", ".join(f"{v:.3f}" for v in t.detach().flatten().tolist()) + "]"


def check_group_relative_advantages(group_relative_advantages) -> list[Check]:
    """Standardize within the group: subtract the mean, divide by the std."""
    checks = []

    # mean 0.5, sample std sqrt(1/3) = 0.577, so each reward is 0.5 / 0.577 = 0.866 from the mean
    adv = group_relative_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    expected = torch.tensor([0.866, -0.866, 0.866, -0.866])
    name = "rewards [1, 0, 1, 0] give [0.866, -0.866, 0.866, -0.866] (mean 0.5, std 0.577)"
    if adv.shape == expected.shape and torch.allclose(adv, expected, atol=1e-3):
        checks.append(ok(name))
    else:
        # [1, -1, 1, -1] is what the n-divisor (population) std gives
        if adv.shape == expected.shape and torch.allclose(adv, torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-3):
            nudge = "use rewards.std(), which divides by n - 1"
        else:
            nudge = "subtract rewards.mean(), then divide by rewards.std() + eps"
        checks.append(bad(name, f"expected {_fmt(expected)}, got {_fmt(adv)} ({nudge})"))

    # Whatever the group, the advantages come out centered with unit spread
    adv = group_relative_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    mean, std = float(adv.mean()), float(adv.std())
    name = "advantages have mean 0 and std 1 within a group: rewards [1, 0, 0, 0, 1, 0, 0, 0]"
    if abs(mean) < 1e-4 and abs(std - 1.0) < 1e-4:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"got mean {mean:.4f} and std {std:.4f} for {_fmt(adv)}"))

    # All tied: nobody is better than the baseline, so there is no signal (and no 0/0)
    adv = group_relative_advantages(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    name = "a group where every reward ties gives all-zero advantages, not NaN: [1, 1, 1, 1]"
    if not torch.isnan(adv).any() and torch.allclose(adv, torch.zeros(4), atol=1e-6):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"got {adv.tolist()} (add eps to the std so 0 / 0 cannot happen)"))
    return checks
