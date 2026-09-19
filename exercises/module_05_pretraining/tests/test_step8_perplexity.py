"""Step 8: loss_to_perplexity_and_bits()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _pair_close(got, expected: tuple[float, float]) -> bool:
    """True if `got` is a (perplexity, bits) pair within 1e-3 of `expected`."""
    try:
        ppl, bits = got
        return math.isclose(ppl, expected[0], abs_tol=1e-3) and math.isclose(bits, expected[1], abs_tol=1e-3)
    except (TypeError, ValueError):
        return False


def _show(got) -> str:
    """Format a (perplexity, bits) pair for a failure message."""
    try:
        ppl, bits = got
        return f"perplexity {float(ppl):.4f}, bits {float(bits):.4f}"
    except (TypeError, ValueError):
        return repr(got)


def check_loss_to_perplexity_and_bits(loss_to_perplexity_and_bits) -> list[Check]:
    """Two readouts of the same loss: exp(loss) and loss / ln 2."""
    checks = []

    # Uniform guessing over 65 characters: loss ln 65, perplexity 65, log2 65 bits
    got = loss_to_perplexity_and_bits(math.log(65))
    swapped = _pair_close(got, (math.log2(65), 65.0))
    checks.append(
        ok("uniform over 65 characters: loss ln 65 gives perplexity 65 and 6.0224 bits")
        if _pair_close(got, (65.0, math.log2(65)))
        else bad("uniform over 65 characters: loss ln 65 gives perplexity 65 and 6.0224 bits",
                 f"got {_show(got)} "
                 + ("(return (perplexity, bits) in that order)" if swapped
                    else "(perplexity is exp(loss); bits is loss / ln 2)"))
    )

    # A perfect model is never surprised
    got = loss_to_perplexity_and_bits(0.0)
    checks.append(
        ok("a loss of 0 (a perfect model) gives perplexity 1 and 0 bits")
        if _pair_close(got, (1.0, 0.0))
        else bad("a loss of 0 (a perfect model) gives perplexity 1 and 0 bits", f"got {_show(got)}")
    )

    # One nat: perplexity e and 1 / ln 2 bits
    got = loss_to_perplexity_and_bits(1.0)
    checks.append(
        ok("a loss of 1 nat gives perplexity e = 2.7183 and 1 / ln 2 = 1.4427 bits")
        if _pair_close(got, (math.e, 1 / math.log(2)))
        else bad("a loss of 1 nat gives perplexity e = 2.7183 and 1 / ln 2 = 1.4427 bits",
                 f"got {_show(got)} (the loss is in nats: use exp for perplexity, divide by ln 2 for bits)")
    )
    return checks
