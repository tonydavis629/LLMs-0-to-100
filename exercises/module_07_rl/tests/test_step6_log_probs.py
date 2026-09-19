"""Step 6: gather_token_log_probs()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def _fmt(t: torch.Tensor) -> str:
    """Show a tensor as a short list with 3 decimals."""
    return "[" + ", ".join(f"{v:.3f}" for v in t.detach().flatten().tolist()) + "]"


def check_gather_token_log_probs(gather_token_log_probs) -> list[Check]:
    """log_softmax over the vocabulary, then pick the taken token at each position."""
    checks = []

    # Row 0: equal logits over 3 tokens, so every token has probability 1/3.
    # Row 1: logits [0, ln 2, 0] give probabilities [1/4, 1/2, 1/4].
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, math.log(2.0), 0.0]])
    got = gather_token_log_probs(logits, torch.tensor([2, 1]))
    expected = torch.tensor([math.log(1 / 3), math.log(1 / 2)])
    name = "logits [[0, 0, 0], [0, ln 2, 0]] with targets [2, 1] give [ln 1/3, ln 1/2] = [-1.099, -0.693]"
    if got.shape == expected.shape and torch.allclose(got, expected, atol=1e-4):
        checks.append(ok(name))
    else:
        # Point at the most likely slip
        if got.shape != expected.shape:
            nudge = ".squeeze(-1) drops the gathered dimension"
        elif bool((got > 0).all()):
            nudge = "those are probabilities: take the log with F.log_softmax"
        else:
            nudge = "log_softmax over the vocabulary, dim=-1"
        checks.append(bad(name, f"expected {_fmt(expected)}, got {_fmt(got)} with shape {tuple(got.shape)} ({nudge})"))

    # A realistic shape: 5 positions over the 69-token vocabulary
    torch.manual_seed(0)
    logits = torch.randn(5, 69)
    targets = torch.randint(0, 69, (5,))
    got = gather_token_log_probs(logits, targets)
    name = "returns one log-prob per position: logits (5, 69) and targets (5,) give shape (5,)"
    if tuple(got.shape) == (5,):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected shape (5,), got {tuple(got.shape)} "
                                "(.squeeze(-1) drops the gathered dimension)"))

    # Cross-entropy is exactly the negative log-prob of the target token
    expected = -F.cross_entropy(logits, targets, reduction="none")
    name = "matches -F.cross_entropy(logits, targets, reduction='none') on random (5, 69) logits"
    if got.numel() == 5 and torch.allclose(got.flatten(), expected, atol=1e-4):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected {_fmt(expected)}, got {_fmt(got)}"))
    return checks
