"""Step 3: decode_tokens_per_second()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _close(got, expected: float, rel: float = 1e-6) -> bool:
    """True if got is a number within a relative tolerance of expected."""
    return isinstance(got, (int, float)) and abs(got - expected) <= rel * abs(expected)


def _show(x) -> str:
    """Print a rate to four significant figures, anything else as-is."""
    if x is None:
        return "None (nothing came back: is the `return` missing?)"
    return f"{x:.4g} tokens/s" if isinstance(x, (int, float)) else repr(x)


def check_decode_tokens_per_second(decode_tokens_per_second) -> list[Check]:
    """Memory bandwidth divided by the bytes read for every token."""
    checks = []

    # A MacBook Air (100 GB/s) reading a 4 GB model for every token
    got = decode_tokens_per_second(100e9, 4e9)
    name = "100 GB/s of bandwidth reading 4 GB per token allows 25 tokens/s"
    if _close(got, 25):
        checks.append(ok(name))
    else:
        nudge = " (upside down: bandwidth goes on top)" if _close(got, 0.04) else ""
        checks.append(bad(name, f"expected {_show(25)}\ngot      {_show(got)}{nudge}"))

    # The lecture's H100 number for fp16 Llama 3 8B (8.03B parameters x 2 bytes)
    got = decode_tokens_per_second(3.35e12, 16.06e9)
    name = "an H100 (3,350 GB/s) reading fp16 Llama-3-8B (16.06 GB) allows about 208.6 tokens/s"
    checks.append(ok(name) if _close(got, 208.59, rel=1e-3) else bad(name, f"expected about {_show(208.6)}\ngot      {_show(got)}"))

    # Double the bandwidth, double the speed; double the bytes, half the speed
    base = decode_tokens_per_second(1e12, 1e9)
    faster = decode_tokens_per_second(2e12, 1e9)
    heavier = decode_tokens_per_second(1e12, 2e9)
    name = "twice the bandwidth doubles the speed, twice the bytes per token halves it"
    if all(isinstance(v, (int, float)) for v in (base, faster, heavier)) and _close(faster, 2 * base) and _close(heavier, base / 2):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"1 TB/s and 1 GB gave {_show(base)}, 2 TB/s gave {_show(faster)}, 2 GB gave {_show(heavier)}"))
    return checks
