"""Step 7: sft_train_step()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from tests.check import Check, bad, ok

# One SGD step (lr = 1) on the toy model below, worked out by hand. The logits
# start uniform (1/4 each), so the gradient at a position is (1/4 - one_hot(target))
# divided by the 2 unmasked positions. Row 1 (input of target 2) and row 2 (input
# of target 3) move against that gradient; rows 0 and 3 are never touched.
EXPECTED_WEIGHT = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [-0.125, -0.125, 0.375, -0.125],
    [-0.125, -0.125, -0.125, 0.375],
    [0.0, 0.0, 0.0, 0.0],
])


def _reference_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """PyTorch's masked cross-entropy, used in place of your Step 3 while testing this step."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def _one_step(sft_train_step, stale_grad: float | None = None):
    """Run your sft_train_step once on a 4-token toy model and return (loss, weight after)."""
    # The toy "language model": an embedding table whose row for token t is the logits after t.
    # All zeros means every next token gets probability 1/4.
    model = nn.Embedding(4, 4)
    nn.init.zeros_(model.weight)
    if stale_grad is not None:
        model.weight.grad = torch.full((4, 4), stale_grad)  # left over from a previous step
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[-100, 2, 3]])  # position 0 is prompt, so it is masked

    # sft_train_step calls your masked_cross_entropy() from Step 3. Swap in PyTorch's
    # version for the test so this step is graded on its own.
    with mock.patch.dict(sft_train_step.__globals__, {"masked_cross_entropy": _reference_loss}):
        loss = sft_train_step(model, optimizer, x, y, grad_clip=1.0)
    return loss, model.weight.detach()


def check_sft_train_step(sft_train_step) -> list[Check]:
    """Zero the old gradients, backpropagate the new loss, and step, on a toy model small enough to do by hand."""
    checks = []

    loss, weight = _one_step(sft_train_step)
    checks.append(
        ok("returns the batch loss as a float: uniform logits over 4 tokens give ln 4 = 1.3863")
        if isinstance(loss, float) and abs(loss - math.log(4)) < 1e-4
        else bad("returns the batch loss as a float: uniform logits over 4 tokens give ln 4 = 1.3863",
                 f"got {loss!r}")
    )

    moved = not torch.equal(weight, torch.zeros(4, 4))
    checks.append(
        ok("one SGD step (lr=1) on a zero embedding moves row 1 to [-0.125, -0.125, 0.375, -0.125]")
        if torch.allclose(weight, EXPECTED_WEIGHT, atol=1e-5)
        else bad("one SGD step (lr=1) on a zero embedding moves row 1 to [-0.125, -0.125, 0.375, -0.125]",
                 f"got row 1 = {[round(v, 4) for v in weight[1].tolist()]}"
                 + ("" if moved else " (nothing moved: call loss.backward(), and zero the gradients "
                                     "before it, not after)"))
    )

    # A gradient left over from the previous batch must be cleared, not added to
    _, weight = _one_step(sft_train_step, stale_grad=100.0)
    checks.append(
        ok("clears the previous step's gradients first (a leftover .grad of 100 changes nothing)")
        if torch.allclose(weight, EXPECTED_WEIGHT, atol=1e-5)
        else bad("clears the previous step's gradients first (a leftover .grad of 100 changes nothing)",
                 f"got row 1 = {[round(v, 4) for v in weight[1].tolist()]} "
                 "(call optimizer.zero_grad(set_to_none=True) before loss.backward())")
    )
    return checks


def check_sft_training(results: dict) -> list[Check]:
    """After finetuning, the masked loss on the instruction data should have collapsed."""
    start, end = results["losses"][0], results["losses"][-1]
    return [
        ok(f"finetuning drives the masked loss below 0.5 (from {start:.2f} at step 0)")
        if end < 0.5
        else bad(f"finetuning drives the masked loss below 0.5 (from {start:.2f} at step 0)",
                 f"final loss {end:.4f}")
    ]
