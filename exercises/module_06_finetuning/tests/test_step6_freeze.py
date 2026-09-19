"""Step 6: freeze_base_param()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


def check_freeze_base_param(freeze_base_param) -> list[Check]:
    """Freezing turns off gradients for that one parameter, in place, and leaves its values alone."""
    checks = []
    torch.manual_seed(0)
    layer = nn.Linear(3, 2)
    before = layer.weight.detach().clone()
    freeze_base_param(layer.weight)

    checks.append(
        ok("sets requires_grad to False on the parameter it is given")
        if layer.weight.requires_grad is False
        else bad("sets requires_grad to False on the parameter it is given",
                 "layer.weight.requires_grad is still True (set p.requires_grad = False on p itself)")
    )
    checks.append(
        ok("leaves the parameter's values unchanged")
        if torch.equal(layer.weight.detach(), before)
        else bad("leaves the parameter's values unchanged", "the weight values changed")
    )

    # After a backward pass the frozen weight gets no gradient; the bias (not frozen) still does
    layer(torch.ones(1, 3)).sum().backward()
    checks.append(
        ok("backward() then skips it: weight.grad stays None, the unfrozen bias still gets one")
        if layer.weight.grad is None and layer.bias.grad is not None
        else bad("backward() then skips it: weight.grad stays None, the unfrozen bias still gets one",
                 f"weight.grad is {'None' if layer.weight.grad is None else 'set'}, "
                 f"bias.grad is {'None' if layer.bias.grad is None else 'set'}")
    )
    return checks


def check_freeze_on_model(model: nn.Module) -> list[Check]:
    """On the real model: every base tensor frozen, every LoRA A and B still trainable."""
    base = [p for name, p in model.named_parameters() if not name.endswith((".A", ".B"))]
    lora = [p for name, p in model.named_parameters() if name.endswith((".A", ".B"))]
    frozen = sum(not p.requires_grad for p in base)
    training = sum(p.requires_grad for p in lora)
    name = (f"real model: all {len(base)} base tensors are frozen and all {len(lora)} LoRA tensors "
            "(A and B) still train")
    return [
        ok(name)
        if frozen == len(base) and training == len(lora)
        else bad(name, f"{frozen} of {len(base)} base tensors frozen, "
                       f"{training} of {len(lora)} LoRA tensors trainable")
    ]
