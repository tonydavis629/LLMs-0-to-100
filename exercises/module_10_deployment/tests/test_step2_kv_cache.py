"""Step 2: kv_cache_bytes_per_token()

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


def check_kv_cache_bytes_per_token(kv_cache_bytes_per_token) -> list[Check]:
    """One key and one value vector, per layer, per KV head, per token."""
    checks = []

    # Small enough to multiply in your head: 2 (K and V) x 2 layers x 1 head x 4 numbers x 2 bytes
    got = kv_cache_bytes_per_token(2, 1, 4, 2)
    name = "2 layers, 1 KV head, head_dim 4, 2 bytes each: 2 x 2 x 1 x 4 x 2 = 32 bytes"
    if _close(got, 32):
        checks.append(ok(name))
    else:
        nudge = " (did you forget the 2 for keys AND values?)" if _close(got, 16) else ""
        checks.append(bad(name, f"expected {_show(32)}\ngot      {_show(got)}{nudge}"))

    # The real Llama 3 8B shape from data/models.json, at fp16
    got = kv_cache_bytes_per_token(32, 8, 128, 2)
    name = "Llama-3-8B (32 layers, 8 KV heads, head_dim 128, fp16) is 131,072 bytes per token"
    checks.append(ok(name) if _close(got, 131_072) else bad(name, f"expected {_show(131_072)}\ngot      {_show(got)}"))

    # Grouped-query attention: the same model with 32 KV heads would need 4x the cache
    mha = kv_cache_bytes_per_token(32, 32, 128, 2)
    gqa = kv_cache_bytes_per_token(32, 8, 128, 2)
    name = "grouped-query attention: 8 KV heads instead of 32 makes the cache 4x smaller"
    if isinstance(mha, (int, float)) and isinstance(gqa, (int, float)) and gqa and _close(mha / gqa, 4):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"32 heads gave {_show(mha)}, 8 heads gave {_show(gqa)} (the cache should grow with n_kv_heads)"))
    return checks
