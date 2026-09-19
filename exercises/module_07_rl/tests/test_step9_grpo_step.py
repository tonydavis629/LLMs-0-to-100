"""Step 9: grpo_step()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok

# A tiny batch for a tiny layer: 2 inputs -> 1 output
_X = torch.tensor([[1.0, 2.0], [3.0, -1.0]])


def _fresh_layer():
    """The same nn.Linear(2, 1) every time, plus its SGD optimizer and a loss on _X."""
    torch.manual_seed(0)
    layer = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss = layer(_X).pow(2).mean()
    return layer, optimizer, loss


def _reference_step(grad_clip: float) -> list[torch.Tensor]:
    """Where one correct step puts the layer: zero_grad, backward, clip, step."""
    layer, optimizer, loss = _fresh_layer()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(layer.parameters(), grad_clip)
    optimizer.step()
    return [p.detach().clone() for p in layer.parameters()]


def _params(layer) -> list[torch.Tensor]:
    """A snapshot of the layer's parameters."""
    return [p.detach().clone() for p in layer.parameters()]


def _matches(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    """True if two parameter snapshots agree."""
    return all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))


def _weight(params: list[torch.Tensor]) -> str:
    """The layer's two weights, rounded for display."""
    return "[" + ", ".join(f"{v:.4f}" for v in params[0].flatten().tolist()) + "]"


def check_grpo_step(grpo_step) -> list[Check]:
    """Zero the gradients, backpropagate, then the provided clip and step."""
    checks = []

    # One ordinary step on a fresh layer
    layer, optimizer, loss = _fresh_layer()
    before = _params(layer)
    grpo_step(optimizer, loss, layer, 1.0)
    after = _params(layer)
    name = "one step moves an nn.Linear layer exactly as zero_grad, backward, clip, step does"
    if _matches(after, _reference_step(1.0)):
        checks.append(ok(name))
    elif _matches(after, before):
        checks.append(bad(name, "the parameters did not move (call loss.backward() after zeroing the gradients)"))
    else:
        checks.append(bad(name, f"expected weights {_weight(_reference_step(1.0))}, got {_weight(after)}"))

    # Leftover gradients from the previous step must not leak into this one.
    # grad_clip is huge here so clipping cannot hide an accumulated gradient.
    layer, optimizer, loss = _fresh_layer()
    for p in layer.parameters():
        p.grad = torch.full_like(p, 100.0)
    grpo_step(optimizer, loss, layer, 1e6)
    name = "clears stale gradients first: a leftover .grad of 100 is not added to this step's"
    if _matches(_params(layer), _reference_step(1e6)):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected weights {_weight(_reference_step(1e6))}, got {_weight(_params(layer))} "
                                "(call optimizer.zero_grad(set_to_none=True) before loss.backward())"))
    return checks
