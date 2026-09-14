"""Extra credit: SGD.step()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


def check_sgd_step(SGD) -> list[Check]:
    """Your optimizer against a hand example and against torch.optim.SGD."""
    checks = []

    # p = [1, 2], grad = [0.5, -1], lr = 0.1  ->  [0.95, 2.1]
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.5, -1.0])
    before = p
    SGD([p], lr=0.1).step()
    checks.append(
        ok("one step: p=[1,2], grad=[0.5,-1], lr=0.1 gives [0.95, 2.1]")
        if torch.allclose(p.detach(), torch.tensor([0.95, 2.1]), atol=1e-6)
        else bad("one step: p=[1,2], grad=[0.5,-1], lr=0.1 gives [0.95, 2.1]", f"got {p.detach().tolist()}")
    )
    checks.append(
        ok("updates the parameter tensor in place (the model keeps pointing at it)")
        if p is before and p.requires_grad
        else bad("updates the parameter tensor in place (the model keeps pointing at it)",
                 "the parameter was replaced or lost requires_grad; use `p -= ...` under torch.no_grad()")
    )

    # Same model, same gradients: one step from each optimizer must agree
    def one_step(make_opt):
        torch.manual_seed(0)
        model = nn.Linear(2, 1)
        opt = make_opt(model.parameters())
        opt.zero_grad()
        model(torch.tensor([[1.0, 2.0], [3.0, -1.0]])).pow(2).mean().backward()
        opt.step()
        return [q.detach().clone() for q in model.parameters()]

    theirs = one_step(lambda ps: torch.optim.SGD(ps, lr=0.1))
    yours = one_step(lambda ps: SGD(ps, lr=0.1))
    checks.append(
        ok("matches torch.optim.SGD after one step on an nn.Linear layer")
        if all(torch.allclose(a, b, atol=1e-6) for a, b in zip(theirs, yours))
        else bad("matches torch.optim.SGD after one step on an nn.Linear layer",
                 f"torch: {[t.flatten().tolist() for t in theirs]}; yours: {[t.flatten().tolist() for t in yours]}")
    )
    return checks


def check_sgd_training(results: dict) -> list[Check]:
    """The MLP trained with your optimizer should learn XOR just as well."""
    correct, total = results["ec_acc"]
    pct = 100 * correct / total
    return [
        ok("the MLP trained with your optimizer reaches at least 95% on XOR")
        if pct >= 95
        else bad("the MLP trained with your optimizer reaches at least 95% on XOR",
                 f"got {correct}/{total} ({pct:.1f}%)")
    ]
