"""Step 5: clip_loss()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_clip_loss(clip_loss) -> list[Check]:
    """Cross-entropy toward the diagonal, over rows and over columns, averaged."""
    checks = []

    # Every entry equal: each row and each column is a 4-way coin toss, ln 4 either way
    got = float(clip_loss(torch.zeros(4, 4)))
    name = "a uniform 4 x 4 matrix (no idea which caption matches) gives ln 4 = 1.386"
    if math.isclose(got, math.log(4), abs_tol=1e-4):
        checks.append(ok(name))
    elif math.isclose(got, 2 * math.log(4), abs_tol=1e-4):
        checks.append(bad(name, f"expected 1.3863, got {got:.4f} (did you add the two directions without halving?)"))
    else:
        checks.append(bad(name, f"expected 1.3863, got {got:.4f}"))

    # A strong diagonal: every image picks its own caption, and every caption its own image
    got = float(clip_loss(10 * torch.eye(3)))
    name = "a strong diagonal, 10 * identity(3), gives nearly zero loss (about 0.0001)"
    checks.append(
        ok(name)
        if 0 <= got < 1e-3
        else bad(name, f"expected about 0.0001, got {got:.4f} (the targets are the diagonal, 0..B-1)")
    )

    # Row-wise cross-entropy (image -> text) and column-wise (text -> image) differ on
    # an uneven matrix; CLIP averages them, so the loss of L and of L.t() must agree
    logits = torch.tensor([[2.0, 1.0, 0.0],
                           [0.0, 3.0, 0.0],
                           [2.0, 2.0, 1.0]])
    forward, backward = float(clip_loss(logits)), float(clip_loss(logits.t()))
    name = "is symmetric: an uneven 3 x 3 matrix and its transpose give the same loss"
    checks.append(
        ok(name)
        if math.isclose(forward, backward, abs_tol=1e-5)
        else bad(name, f"got {forward:.4f} for the matrix and {backward:.4f} for its transpose "
                       "(average the image->text rows and the text->image columns)")
    )

    # Reference: the two directions with torch's own cross-entropy
    torch.manual_seed(0)
    logits = torch.randn(5, 5) * 3
    labels = torch.arange(5)
    expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    got = float(clip_loss(logits))
    name = "equals the mean of F.cross_entropy over rows and over columns (random 5 x 5 logits)"
    checks.append(
        ok(name)
        if math.isclose(got, float(expected), abs_tol=1e-4)
        else bad(name, f"expected {float(expected):.4f}, got {got:.4f}")
    )
    return checks


def check_clip_training(results: dict) -> list[Check]:
    """After contrastive training, most held-out images should find their own caption."""
    before, after = results["clip_acc_before"], results["clip_acc_after"]
    name = "held-out retrieval accuracy climbs above 50% after training (chance is 1/60)"
    return [
        ok(name)
        if after >= 0.5
        else bad(name, f"went from {before:.1%} before training to {after:.1%} after")
    ]
