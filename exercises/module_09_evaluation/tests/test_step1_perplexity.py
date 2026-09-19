"""Step 1: perplexity()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _close(name: str, loss: float, got, expected: float) -> Check:
    """Pass if the perplexity for `loss` is within 0.1% of the expected value."""
    if isinstance(got, (int, float)) and math.isclose(got, expected, rel_tol=1e-3):
        return ok(name)
    shown = f"{got:.4f}" if isinstance(got, (int, float)) else repr(got)
    detail = f"expected {expected:.4f}, got {shown}"
    # The most common slip: treating the loss as bits and raising 2 to it
    if isinstance(got, (int, float)) and math.isclose(got, 2 ** loss, rel_tol=1e-3):
        detail += " (the loss is in nats, so the base is e: use math.exp, not 2 **)"
    return bad(name, detail)


def check_perplexity(perplexity) -> list[Check]:
    """Perplexity is e raised to the average loss in nats."""
    checks = []

    # A model that is never surprised has zero loss and chooses among 1 option
    checks.append(_close("a loss of 0 nats (never surprised) gives perplexity 1.0",
                         0.0, perplexity(0.0), 1.0))

    # -ln(1/2) = ln 2 per token: a coin flip between two tokens
    checks.append(_close("a loss of ln 2 = 0.693 nats (a coin flip per token) gives perplexity 2.0",
                         math.log(2), perplexity(math.log(2)), 2.0))

    # Module 1's uniform model over 27 characters: every character costs ln 27
    checks.append(_close("a loss of ln 27 = 3.296 nats (uniform over 27 characters) gives perplexity 27.0",
                         math.log(27), perplexity(math.log(27)), 27.0))
    return checks
