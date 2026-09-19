"""Step 10: mean_reward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_mean_reward(mean_reward) -> list[Check]:
    """The average reward of a group, as a plain Python number."""
    checks = []

    got = mean_reward(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    checks.append(
        ok("rewards [1, 0, 1, 0] average to 0.5")
        if abs(float(got) - 0.5) < 1e-6
        else bad("rewards [1, 0, 1, 0] average to 0.5", f"got {float(got):.4f}")
    )

    # The runner stores these in Python lists for the reward curve
    checks.append(
        ok("returns a Python float, not a tensor")
        if type(got) is float
        else bad("returns a Python float, not a tensor", f"got {type(got).__name__} {got!r} (.item() converts it)")
    )

    got = mean_reward(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))
    checks.append(
        ok("a group of 8 with 3 verified averages 3 / 8 = 0.375")
        if abs(float(got) - 0.375) < 1e-6
        else bad("a group of 8 with 3 verified averages 3 / 8 = 0.375", f"got {float(got):.4f}")
    )
    return checks
