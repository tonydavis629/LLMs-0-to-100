"""Step 8: count_trainable_params()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


class _ToyLoRA(nn.Module):
    """A frozen 4 -> 3 layer (12 + 3 = 15 numbers) with a rank-2 adapter: A is 2 x 4, B is 3 x 2."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 3)
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False
        self.A = nn.Parameter(torch.zeros(2, 4))
        self.B = nn.Parameter(torch.zeros(3, 2))


def _count_check(name: str, got, expected: int, tensors: int, everything: int) -> Check:
    """Compare a count, with a nudge when it matches a common mistake."""
    if got == expected:
        return ok(name)
    nudge = ""
    if got == tensors and tensors != expected:
        nudge = " (that is the number of tensors: add up p.numel() instead)"
    elif got == everything and everything != expected:
        nudge = " (that counts the frozen parameters too: keep only p.requires_grad)"
    return bad(name, f"expected {expected}, got {got!r}{nudge}")


def check_count_trainable_params(count_trainable_params) -> list[Check]:
    """Count the numbers that will train (elements, not tensors), skipping frozen ones."""
    checks = []

    # Nothing frozen: nn.Linear(3, 2) has a 2 x 3 weight and 2 biases
    checks.append(_count_check("counts every element: nn.Linear(3, 2) has 2*3 + 2 = 8 trainable numbers",
                               count_trainable_params(nn.Linear(3, 2)), 8, tensors=2, everything=8))

    # The LoRA case: only A and B count
    checks.append(_count_check("skips frozen tensors: a rank-2 adapter on a frozen 4 -> 3 layer gives 2*4 + 3*2 = 14",
                               count_trainable_params(_ToyLoRA()), 14, tensors=2, everything=29))

    # Everything frozen: nothing left to train
    frozen = nn.Linear(3, 2).requires_grad_(False)
    checks.append(_count_check("a fully frozen layer has 0 trainable parameters",
                               count_trainable_params(frozen), 0, tensors=0, everything=8))
    return checks


def check_count_on_model(count_trainable_params, model: nn.Module) -> list[Check]:
    """On the real model, the trainable numbers are exactly the LoRA A and B matrices."""
    expected = sum(p.numel() for name, p in model.named_parameters() if name.endswith((".A", ".B")))
    total = sum(p.numel() for p in model.parameters())
    tensors = sum(p.requires_grad for p in model.parameters())
    return [_count_check(f"real model: the count is exactly the LoRA A and B matrices, {expected:,} numbers",
                         count_trainable_params(model), expected, tensors=tensors, everything=total)]
