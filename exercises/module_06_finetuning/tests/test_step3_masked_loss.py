"""Step 3: masked_cross_entropy()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_masked_cross_entropy(masked_cross_entropy) -> list[Check]:
    """Cross-entropy averaged over the response positions only; -100 positions count for nothing."""
    checks = []

    # Uniform logits: every token gets probability 1/4, so each position costs ln 4
    logits = torch.zeros(1, 3, 4)
    targets = torch.tensor([[0, 2, 3]])
    loss = float(masked_cross_entropy(logits, targets))
    checks.append(
        ok("uniform logits over 4 tokens cost ln 4 = 1.3863 per position")
        if abs(loss - math.log(4)) < 1e-4
        else bad("uniform logits over 4 tokens cost ln 4 = 1.3863 per position", f"got {loss:.4f}")
    )

    # One real position (logits [2, 0, 0, 0], target 0) and one masked position.
    # The answer is the real position alone: ln(e^2 + 3) - 2 = 0.3408
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [5.0, -5.0, 1.0, 0.0]]])
    targets = torch.tensor([[0, -100]])
    expected = math.log(math.exp(2) + 3) - 2
    loss = float(masked_cross_entropy(logits, targets))
    checks.append(
        ok("a -100 position adds nothing: [real, masked] costs the real position alone, 0.3408")
        if abs(loss - expected) < 1e-4
        else bad("a -100 position adds nothing: [real, masked] costs the real position alone, 0.3408",
                 f"got {loss:.4f}"
                 + (" (half the answer: the masked position was counted in the average)"
                    if abs(loss - expected / 2) < 1e-4 else ""))
    )

    # Cross-check against PyTorch on a random batch, averaging only the unmasked rows
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 69)
    targets = torch.randint(0, 69, (2, 6))
    targets[:, :3] = -100  # the first three positions of each row are prompt
    keep = targets != -100
    expected = float(F.cross_entropy(logits[keep], targets[keep]))
    loss = float(masked_cross_entropy(logits, targets))
    checks.append(
        ok("matches F.cross_entropy over just the unmasked positions of a random (2, 6, 69) batch")
        if abs(loss - expected) < 1e-4
        else bad("matches F.cross_entropy over just the unmasked positions of a random (2, 6, 69) batch",
                 f"expected {expected:.4f}, got {loss:.4f}")
    )

    # Training calls loss.backward(), so the loss must stay a tensor in the graph
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    out = masked_cross_entropy(logits, torch.tensor([[-100, 1, 2]]))
    is_scalar = isinstance(out, torch.Tensor) and out.dim() == 0 and out.requires_grad
    checks.append(
        ok("returns a scalar tensor that can backpropagate (not a Python float)")
        if is_scalar
        else bad("returns a scalar tensor that can backpropagate (not a Python float)",
                 f"got {type(out).__name__}"
                 + (f" with shape {tuple(out.shape)}" if isinstance(out, torch.Tensor) else "")
                 + " (return the F.cross_entropy result directly, without .item())")
    )
    return checks
