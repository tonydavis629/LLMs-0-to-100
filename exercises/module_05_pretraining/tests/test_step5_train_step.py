"""Step 5: train_step()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tests.check import Check, bad, ok

VOCAB = 5  # a 5-token vocabulary keeps every tensor tiny


def _bigram(weights: torch.Tensor) -> nn.Embedding:
    """A bigram model: row i of the table is the logits for the token after token i.

    nn.Embedding maps a (batch, time) tensor of IDs to (batch, time, VOCAB)
    logits, the same shapes TinyGPT uses, so train_step() can train it.
    """
    return nn.Embedding.from_pretrained(weights.clone(), freeze=False)


def _reference_step(weights: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One correct training step, written out in full, for comparison."""
    model = _bigram(weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return model.weight.detach()


def check_train_step(train_step) -> list[Check]:
    """Zero the gradients, backpropagate, step: compared against the same step done by hand."""
    checks = []
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (2, 3), generator=gen)
    y = torch.randint(0, VOCAB, (2, 3), generator=gen)

    # A table of zeros predicts every token with probability 1/5
    model = _bigram(torch.zeros(VOCAB, VOCAB))
    loss = train_step(model, torch.optim.SGD(model.parameters(), lr=0.5), x, y)
    checks.append(
        ok("returns the batch loss as a float: ln 5 = 1.6094 for a table of zeros")
        if isinstance(loss, float) and math.isclose(loss, math.log(VOCAB), abs_tol=1e-4)
        else bad("returns the batch loss as a float: ln 5 = 1.6094 for a table of zeros",
                 f"got {loss!r}")
    )

    # One SGD step from random weights must land exactly where the hand-written step does
    start = torch.randn(VOCAB, VOCAB, generator=gen)
    expected = _reference_step(start, x, y)
    model = _bigram(start)
    train_step(model, torch.optim.SGD(model.parameters(), lr=0.5), x, y)
    moved = not torch.allclose(model.weight.detach(), start)
    checks.append(
        ok("one SGD step (lr 0.5) moves the weights exactly as backward() + step() should")
        if torch.allclose(model.weight.detach(), expected, atol=1e-5)
        else bad("one SGD step (lr 0.5) moves the weights exactly as backward() + step() should",
                 "the weights moved to the wrong place" if moved else
                 "the weights did not move (call loss.backward() after zeroing the gradients)")
    )

    # Leave junk in .grad first: a correct step clears it before backpropagating
    model = _bigram(start)
    model.weight.grad = torch.full((VOCAB, VOCAB), 100.0)
    train_step(model, torch.optim.SGD(model.parameters(), lr=0.5), x, y)
    checks.append(
        ok("clears old gradients first: a leftover .grad of 100 does not leak into the step")
        if torch.allclose(model.weight.detach(), expected, atol=1e-5)
        else bad("clears old gradients first: a leftover .grad of 100 does not leak into the step",
                 "the leftover gradient changed the update "
                 "(call optimizer.zero_grad() before loss.backward())")
    )
    return checks


def check_overfit(results: dict) -> list[Check]:
    """The --overfit sanity check: one fixed batch should be memorized."""
    losses = results["overfit_losses"]
    return [
        ok("training on one fixed batch drives its loss from about 4.18 to under 0.5")
        if losses[0] > 4.0 and losses[-1] < 0.5
        else bad("training on one fixed batch drives its loss from about 4.18 to under 0.5",
                 f"the loss went from {losses[0]:.4f} to {losses[-1]:.4f}")
    ]
