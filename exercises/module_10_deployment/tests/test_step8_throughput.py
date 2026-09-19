"""Step 8: total_throughput()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _close(got, expected: float) -> bool:
    """True if got is a number within a millionth of expected."""
    return isinstance(got, (int, float)) and abs(got - expected) <= 1e-6 * abs(expected)


def _show(x) -> str:
    """Print a rate to four significant figures, anything else as-is."""
    if x is None:
        return "None (nothing came back: is the `return` missing?)"
    return f"{x:.4g} tokens/s" if isinstance(x, (int, float)) else repr(x)


def check_total_throughput(total_throughput) -> list[Check]:
    """Every token the machine produced, divided by the wall-clock time."""
    checks = []

    # Four requests of 80 tokens each, all finished 2 seconds after the first send
    got = total_throughput([80, 80, 80, 80], 2.0)
    name = "4 requests of 80 tokens in 2.0 s is 320 / 2.0 = 160 tokens/s"
    if _close(got, 160.0):
        checks.append(ok(name))
    else:
        nudge = " (that is one request's rate: add up every request with sum())" if _close(got, 40.0) else ""
        checks.append(bad(name, f"expected {_show(160)}\ngot      {_show(got)}{nudge}"))

    # Replies of different lengths all count
    got = total_throughput([10, 30, 60], 4.0)
    name = "uneven replies of 10, 30, and 60 tokens in 4.0 s give 100 / 4.0 = 25 tokens/s"
    checks.append(ok(name) if _close(got, 25.0) else bad(name, f"expected {_show(25)}\ngot      {_show(got)}"))

    # A single request is just its own rate
    got = total_throughput([120], 1.5)
    name = "a single request of 120 tokens in 1.5 s is 80 tokens/s"
    checks.append(ok(name) if _close(got, 80.0) else bad(name, f"expected {_show(80)}\ngot      {_show(got)}"))
    return checks
