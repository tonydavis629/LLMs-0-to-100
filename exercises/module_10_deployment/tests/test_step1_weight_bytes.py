"""Step 1: model_weight_bytes()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _close(got, expected: float) -> bool:
    """True if got is a number within a millionth of expected."""
    return isinstance(got, (int, float)) and abs(got - expected) <= 1e-6 * abs(expected)


def _show(x) -> str:
    """Print a byte count with thousands separators, anything else as-is."""
    if x is None:
        return "None (nothing came back: is the `return` missing?)"
    return f"{x:,.0f} bytes" if isinstance(x, (int, float)) else repr(x)


def check_model_weight_bytes(model_weight_bytes) -> list[Check]:
    """Parameters times bits, converted from bits to bytes."""
    checks = []

    # The lecture's example: 7B parameters at 16 bits is 112 billion bits, 14 billion bytes
    got = model_weight_bytes(7_000_000_000, 16)
    name = "7B parameters at 16 bits is 14 GB (14,000,000,000 bytes)"
    if _close(got, 14e9):
        checks.append(ok(name))
    else:
        nudge = ""
        if _close(got, 112e9):
            nudge = " (that is the number of bits: divide by 8)"
        elif isinstance(got, (int, float)):
            nudge = " (8 bits make a byte)"
        checks.append(bad(name, f"expected {_show(14e9)}\ngot      {_show(got)}{nudge}"))

    # Quantizing to 4 bits cuts the memory to a quarter of fp16
    got = model_weight_bytes(7_000_000_000, 4)
    name = "the same model at 4 bits is 3.5 GB, a quarter of the fp16 size"
    checks.append(ok(name) if _close(got, 3.5e9) else bad(name, f"expected {_show(3.5e9)}\ngot      {_show(got)}"))

    # At 8 bits every parameter takes exactly one byte
    got = model_weight_bytes(1000, 8)
    name = "1,000 parameters at 8 bits is 1,000 bytes (one byte each)"
    checks.append(ok(name) if _close(got, 1000) else bad(name, f"expected {_show(1000)}\ngot      {_show(got)}"))
    return checks
