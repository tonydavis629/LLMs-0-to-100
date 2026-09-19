"""Step 7: captioning_loss()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_captioning_loss(captioning_loss) -> list[Check]:
    """Average next-token cross-entropy over the response positions only."""
    checks = []

    # Three positions, vocabulary of 2, and the correct next token is always 0.
    # Position 0 is prompt (mask False) and is confidently wrong: it must be ignored.
    # Position 1 gives token 0 probability 0.5, position 2 gives it 0.75.
    logits = torch.tensor([[0.0, 10.0],
                           [0.0, 0.0],
                           [math.log(3.0), 0.0]])
    targets = torch.tensor([0, 0, 0])
    mask = torch.tensor([False, True, True])
    got = float(captioning_loss(logits, targets, mask))
    expected = (-math.log(0.5) - math.log(0.75)) / 2  # 0.4904
    name = "averages -log p(target) over the response only: p = 0.5 and 0.75 give 0.490"
    if math.isclose(got, expected, abs_tol=1e-4):
        checks.append(ok(name))
    elif math.isclose(got, float(F.cross_entropy(logits, targets)), abs_tol=1e-4):
        checks.append(bad(name, f"expected {expected:.4f}, got {got:.4f} "
                                "(the prompt position was scored too: index logits and targets with the mask)"))
    elif math.isclose(got, 2 * expected, abs_tol=1e-4):
        checks.append(bad(name, f"expected {expected:.4f}, got {got:.4f} (average over the positions, do not sum)"))
    else:
        checks.append(bad(name, f"expected {expected:.4f}, got {got:.4f}"))

    # Whatever the model predicts at a masked-out position, the loss must not move
    torch.manual_seed(0)
    logits = torch.randn(6, 10)
    targets = torch.randint(0, 10, (6,))
    mask = torch.tensor([False, False, True, True, True, False])
    before = float(captioning_loss(logits, targets, mask))
    scrambled = logits.clone()
    scrambled[~mask] = torch.randn(3, 10) * 50
    after = float(captioning_loss(scrambled, targets, mask))
    name = "scrambling the logits at masked-out positions leaves the loss unchanged"
    checks.append(
        ok(name)
        if math.isclose(before, after, abs_tol=1e-5)
        else bad(name, f"loss changed from {before:.4f} to {after:.4f}")
    )

    # Reference: torch's cross-entropy told to ignore every non-response position
    reference = F.cross_entropy(logits, targets.masked_fill(~mask, -100), ignore_index=-100)
    name = "matches F.cross_entropy with the prompt positions set to ignore_index (random 6 x 10)"
    checks.append(
        ok(name)
        if math.isclose(before, float(reference), abs_tol=1e-4)
        else bad(name, f"expected {float(reference):.4f}, got {before:.4f}")
    )
    return checks


def check_bridge_training(results: dict) -> list[Check]:
    """Finetuning the projector and the language model should drive the loss down."""
    losses = results["bridge_losses"]
    first = losses[0]
    last = sum(losses[-50:]) / len(losses[-50:])
    name = "bridge training takes the captioning loss from about 9 to under 0.1 (last 50 steps)"
    return [
        ok(name)
        if last < 0.1
        else bad(name, f"started at {first:.3f}, averaged {last:.3f} over the last 50 steps")
    ]
