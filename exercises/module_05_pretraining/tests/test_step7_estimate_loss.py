"""Step 7: estimate_loss()

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
    """A bigram model: row i of the table is the logits for the token after token i."""
    return nn.Embedding.from_pretrained(weights.clone(), freeze=False)


def _reference_losses(model, data, block_size, batch_size, n_batches, seed) -> list[float]:
    """Draw the same batches get_batch() draws, and score each one by hand."""
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=gen)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        with torch.no_grad():
            losses.append(F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)).item())
    return losses


def check_estimate_loss(estimate_loss) -> list[Check]:
    """The loss averaged over several batches, on a tiny model with known answers."""
    checks = []
    data = torch.randint(0, VOCAB, (50,), generator=torch.Generator().manual_seed(0))

    # A table of zeros guesses uniformly, so every batch costs ln 5
    zeros = _bigram(torch.zeros(VOCAB, VOCAB))
    got = estimate_loss(zeros, data, 4, 2, 3, torch.Generator().manual_seed(1))
    checks.append(
        ok("a uniform model scores ln 5 = 1.6094 on every batch, so the average is ln 5")
        if math.isclose(float(got), math.log(VOCAB), abs_tol=1e-4)
        else bad("a uniform model scores ln 5 = 1.6094 on every batch, so the average is ln 5",
                 f"got {float(got):.4f} (add each batch's loss to total with +=, do not replace it)")
    )

    # A random table: each batch scores differently, so the average must use every batch
    table = _bigram(torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(2)))
    got = estimate_loss(table, data, 4, 2, 3, torch.Generator().manual_seed(1))
    losses = _reference_losses(table, data, 4, 2, 3, seed=1)
    expected = sum(losses) / len(losses)
    # Keeping only the last batch's loss is the most likely slip
    only_last = math.isclose(float(got), losses[-1] / len(losses), abs_tol=1e-4)
    checks.append(
        ok("matches the mean of compute_loss(logits, y) over the same 3 seeded batches")
        if math.isclose(float(got), expected, abs_tol=1e-4)
        else bad("matches the mean of compute_loss(logits, y) over the same 3 seeded batches",
                 f"expected {expected:.4f}, got {float(got):.4f} "
                 + ("(only the last batch counted: use += to add every batch)" if only_last
                    else "(score the logits against the targets y)"))
    )

    # The training loop prints this number and stores it in a list
    checks.append(
        ok("returns a plain Python float")
        if isinstance(got, float)
        else bad("returns a plain Python float",
                 f"got {type(got).__name__} (use .item() to turn each batch loss into a float)")
    )
    return checks


def check_pretraining(results: dict) -> list[Check]:
    """After the full run, the model should have learned a lot about Shakespeare."""
    val = results["val_hist"]
    return [
        ok("pretraining lowers the validation loss from about ln 65 = 4.17 to under 2.0")
        if val[0] > 4.0 and val[-1] < 2.0
        else bad("pretraining lowers the validation loss from about ln 65 = 4.17 to under 2.0",
                 f"validation loss went from {val[0]:.4f} to {val[-1]:.4f}")
    ]
