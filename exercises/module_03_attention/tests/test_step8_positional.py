"""Step 8: add_positional_embeddings()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch

from tests.check import Check, bad, ok


def _rounded(t: torch.Tensor) -> list[float]:
    """A tensor row as a short list of 4-decimal numbers."""
    return [round(v, 4) for v in t.tolist()]


def check_add_positional_embeddings(add_positional_embeddings) -> list[Check]:
    """Sine on even dimensions, cosine on odd ones, added on top of the tokens."""
    checks = []

    # All-zero tokens, so the output is the positional encoding alone
    P = add_positional_embeddings(torch.zeros(2, 4))

    name = "position 0 is [sin 0, cos 0, sin 0, cos 0] = [0, 1, 0, 1]"
    checks.append(
        ok(name)
        if tuple(P.shape) == (2, 4) and torch.allclose(P[0], torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-6)
        else bad(name, f"got {_rounded(P[0])} (sine goes in dimensions 0, 2, ...; cosine in 1, 3, ...)")
    )

    # d_model=4: dimensions 0-1 turn at rate 1, dimensions 2-3 at rate 1/10000^(2/4) = 0.01
    expected = torch.tensor([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)])
    name = "position 1, d_model=4 is [sin 1, cos 1, sin 0.01, cos 0.01] = [0.841, 0.540, 0.010, 1.000]"
    if tuple(P.shape) == (2, 4) and torch.allclose(P[1], expected, atol=1e-4):
        checks.append(ok(name))
    else:
        swapped = torch.tensor([math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)])
        nudge = ""
        if tuple(P.shape) == (2, 4) and torch.allclose(P[1], swapped, atol=1e-4):
            nudge = " (sine and cosine are swapped: sine fills P[:, 0::2])"
        checks.append(bad(name, f"got {_rounded(P[1].flatten())}{nudge}"))

    # The formula from the lecture, one entry at a time, for a longer sequence
    seq_len, d_model = 50, 16
    reference = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(d_model // 2):
            angle = pos / 10000 ** (2 * i / d_model)
            reference[pos, 2 * i] = math.sin(angle)
            reference[pos, 2 * i + 1] = math.cos(angle)
    P = add_positional_embeddings(torch.zeros(seq_len, d_model))
    name = "matches sin and cos of pos / 10000^(2i/d_model) for 50 positions, d_model=16"
    if P.shape == reference.shape and torch.allclose(P, reference, atol=1e-4):
        checks.append(ok(name))
    else:
        worst = int((P - reference).abs().max(dim=1).values.argmax()) if P.shape == reference.shape else 0
        checks.append(bad(name, f"position {worst}: expected {_rounded(reference[worst, :4])}...\n"
                                f"got {_rounded(P.flatten()[worst * d_model:worst * d_model + 4])}..."))

    # Filling X directly (instead of P) would overwrite the token embeddings
    X = torch.ones(3, 4)
    out = add_positional_embeddings(X)
    P = add_positional_embeddings(torch.zeros(3, 4))
    name = "adds the encoding on top of the tokens and leaves the input X unchanged"
    checks.append(
        ok(name)
        if torch.equal(X, torch.ones(3, 4)) and torch.allclose(out, 1 + P, atol=1e-6)
        else bad(name, "X was changed or replaced (fill P, the zeros tensor; X + P is returned for you)")
    )
    return checks
