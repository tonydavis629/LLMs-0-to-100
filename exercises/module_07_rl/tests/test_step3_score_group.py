"""Step 3: score_group()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_score_group(score_group) -> list[Check]:
    """One verifier reward per completion, collected into a float tensor."""
    checks = []

    # The values first (a list or a tensor both count here; the next check looks at the type)
    rewards = score_group(["tac", "cat", "tac", "ta"], "tac")
    values = [float(v) for v in rewards]
    name = "scores each completion: ['tac', 'cat', 'tac', 'ta'] against 'tac' gives [1, 0, 1, 0]"
    if values == [1.0, 0.0, 1.0, 0.0]:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected [1.0, 0.0, 1.0, 0.0], got {values}"))

    # The advantages in Step 4 need a float tensor to take a mean and std
    name = "returns a 1-D float tensor with one reward per completion"
    if isinstance(rewards, torch.Tensor) and rewards.dim() == 1 and rewards.dtype.is_floating_point:
        checks.append(ok(name))
    else:
        kind = f"a tensor of dtype {rewards.dtype}, shape {tuple(rewards.shape)}" \
            if isinstance(rewards, torch.Tensor) else f"a {type(rewards).__name__}"
        checks.append(bad(name, f"got {kind} (wrap the list in torch.tensor(...))"))

    # A group where nobody succeeds is all zeros, one per member
    values = [float(v) for v in score_group(["x"] * 8, "tac")]
    name = "a group of 8 where nobody verifies scores 8 zeros"
    if values == [0.0] * 8:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"got {values}"))
    return checks
