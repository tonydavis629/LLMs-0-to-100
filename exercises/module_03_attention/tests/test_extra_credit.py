"""Extra credit: kv_cache_step()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def _rounded(t: torch.Tensor) -> list[float]:
    """A tensor as a flat list of 4-decimal numbers."""
    return [round(v, 4) for v in t.flatten().tolist()]


def check_kv_cache_step(kv_cache_step) -> list[Check]:
    """One new token: project it, grow the cache by one row, attend over the cache."""
    checks = []

    # d_model = d_k = 2 with hand-picked weights; the new token is x = [1, 2]
    W_Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    W_K = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # swaps the two features: k = [2, 1]
    W_V = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # v = [1 + 2, 2] = [3, 2]
    x = torch.tensor([[1.0, 2.0]])

    # With nothing cached, the token can only attend to itself
    empty = torch.zeros(0, 2)
    out, keys, values = kv_cache_step(x, empty, empty, W_Q, W_K, W_V, 2)
    name = "with an empty cache, the output is the token's own value: x=[1,2] gives v = x @ W_V = [3, 2]"
    checks.append(
        ok(name)
        if tuple(out.shape) == (1, 2) and torch.allclose(out, torch.tensor([[3.0, 2.0]]), atol=1e-6)
        else bad(name, f"got {_rounded(out)} (the value comes from W_V)")
    )

    # Two tokens already cached: the new key and value go on the end
    old = torch.tensor([[5.0, 5.0], [6.0, 6.0]])
    out, keys, values = kv_cache_step(x, old, old, W_Q, W_K, W_V, 2)
    name = "appends one row to each cache: new key x @ W_K = [2, 1], new value x @ W_V = [3, 2]"
    shapes_ok = tuple(keys.shape) == (3, 2) and tuple(values.shape) == (3, 2)
    if shapes_ok and torch.allclose(keys[-1], torch.tensor([2.0, 1.0])) and torch.allclose(
        values[-1], torch.tensor([3.0, 2.0])
    ):
        checks.append(ok(name))
    else:
        detail = f"cache shapes {tuple(keys.shape)} and {tuple(values.shape)}, expected (3, 2) and (3, 2)"
        if shapes_ok:
            detail = f"last key {_rounded(keys[-1])}, last value {_rounded(values[-1])}"
        checks.append(bad(name, detail + " (key from W_K, value from W_V)"))

    # A whole sequence, one token at a time, against causal attention computed all at once
    torch.manual_seed(0)
    X = torch.randn(5, 8)
    W_Q, W_K, W_V = torch.randn(8, 4), torch.randn(8, 4), torch.randn(8, 4)
    keys = values = torch.zeros(0, 4)
    rows = []
    for t in range(5):
        out, keys, values = kv_cache_step(X[t:t + 1], keys, values, W_Q, W_K, W_V, 4)
        rows.append(out)
    yours = torch.cat(rows, dim=0)
    reference = F.scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V, is_causal=True)
    name = "5 tokens one at a time match full causal attention recomputed from scratch"
    checks.append(
        ok(name)
        if yours.shape == reference.shape and torch.allclose(yours, reference, atol=1e-4)
        else bad(name, f"token 4: expected {_rounded(reference[4])}\n"
                       f"got {_rounded(yours.flatten()[-4:])} (the query comes from W_Q)")
    )
    return checks
