"""Step 7: latency_stats()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _rate(stats):
    """Pull tokens_per_second out of the result, or None if it is missing."""
    return stats.get("tokens_per_second") if isinstance(stats, dict) else None


def _close(got, expected: float) -> bool:
    """True if got is a number within a millionth of expected."""
    return isinstance(got, (int, float)) and abs(got - expected) <= 1e-6 * abs(expected)


def _show(x) -> str:
    """Print a rate to four significant figures, anything else as-is."""
    if x is None:
        return "None (nothing came back: is the `return` missing?)"
    return f"{x:.4g} tokens/s" if isinstance(x, (int, float)) else repr(x)


def check_latency_stats(latency_stats) -> list[Check]:
    """TTFT is the wait for token one; the decode rate counts only the tokens after it."""
    checks = []

    # Sent at t=10.0, five tokens arrive at 10.5, 10.6, 10.7, 10.8, 10.9
    stats = latency_stats(10.0, [10.5, 10.6, 10.7, 10.8, 10.9])
    name = "returns a dict with both keys, ttft and tokens_per_second"
    if not (isinstance(stats, dict) and {"ttft", "tokens_per_second"} <= stats.keys()):
        got = sorted(stats) if isinstance(stats, dict) else stats
        return [bad(name, f"got {got!r} (return {{\"ttft\": ttft, \"tokens_per_second\": tokens_per_second}})")]
    checks.append(ok(name))

    got = _rate(stats)
    name = "sent at 10.0, 5 tokens at 10.5, 10.6, ..., 10.9: 4 tokens in 0.4 s is 10 tokens/s"
    if _close(got, 10.0):
        checks.append(ok(name))
    else:
        nudge = ""
        if _close(got, 12.5):
            nudge = " (token 1 belongs to prefill: count len(token_times) - 1)"
        elif isinstance(got, (int, float)) and got < 6:
            nudge = " (time the decode from token_times[0], not from start_time)"
        checks.append(bad(name, f"expected {_show(10)}\ngot      {_show(got)}{nudge}"))

    # A slow prefill must not drag the decode rate down: 2 s wait, then 11 tokens 0.05 s apart
    times = [2.0 + 0.05 * i for i in range(11)]
    got = _rate(latency_stats(0.0, times))
    name = "a 2 s wait for the first token, then 11 tokens 0.05 s apart, still decodes at 20 tokens/s"
    if _close(got, 20.0):
        checks.append(ok(name))
    else:
        nudge = ""
        if _close(got, 22.0):
            nudge = " (count only the len(token_times) - 1 tokens after the first)"
        elif isinstance(got, (int, float)) and got < 20:
            nudge = " (the 2 s wait before token 1 is TTFT, not decode time)"
        checks.append(bad(name, f"expected {_show(20)}\ngot      {_show(got)}{nudge}"))
    return checks
