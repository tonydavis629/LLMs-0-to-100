"""Step 4: build_optimizer()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


def _toy_model() -> nn.Module:
    """Two small layers: the first is frozen (the "base"), the second trains (the "adapter")."""
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 2))
    for p in model[0].parameters():
        p.requires_grad = False
    return model


def check_build_optimizer(build_optimizer) -> list[Check]:
    """AdamW over the trainable parameters, and nothing else."""
    checks = []
    model = _toy_model()
    optimizer = build_optimizer(model, 0.01)

    checks.append(
        ok("returns a torch.optim.AdamW optimizer")
        if isinstance(optimizer, torch.optim.AdamW)
        else bad("returns a torch.optim.AdamW optimizer", f"got {type(optimizer).__name__}")
    )

    # Which tensors did the optimizer get? Compare by identity, not by value
    held = [p for group in getattr(optimizer, "param_groups", []) for p in group["params"]]
    trainable = [model[1].weight, model[1].bias]
    same = len(held) == len(trainable) and all(any(p is q for q in held) for p in trainable)
    checks.append(
        ok("holds only the trainable tensors (2 frozen + 2 trainable gives just those 2)")
        if same
        else bad("holds only the trainable tensors (2 frozen + 2 trainable gives just those 2)",
                 f"the optimizer holds {len(held)} tensors, "
                 f"{sum(not p.requires_grad for p in held)} of them frozen "
                 "(keep only the parameters with p.requires_grad)")
    )

    lrs = [group.get("lr") for group in getattr(optimizer, "param_groups", [])]
    checks.append(
        ok("uses the learning rate it is given (lr=0.01)")
        if lrs and all(lr == 0.01 for lr in lrs)
        else bad("uses the learning rate it is given (lr=0.01)", f"got lr={lrs}")
    )
    return checks
