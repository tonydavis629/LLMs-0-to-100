"""Step 7: TinyAttentionLayer.masked_attention() (provided; uses Steps 2, 3, 5 and 6)

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok

# masked_attention() is written for you, so a failure here points back to an earlier step
_WHERE = "(masked_attention() runs your Steps 2, 3, 5 and 6; fix any of those marked INCORRECT)"


def check_masked_attention(TinyAttentionLayer, X: torch.Tensor) -> list[Check]:
    """Causal attention on the 5 exercise tokens, checked against torch's own."""
    checks = []
    layer = TinyAttentionLayer(d_model=X.shape[1], d_k=4)
    output, weights = layer.masked_attention(X)
    n = X.shape[0]

    # Token 0 has no past, so all of its weight lands on itself
    name = "the first token can only attend to itself: its weight row is [1, 0, 0, 0, 0]"
    first = torch.zeros(n)
    first[0] = 1.0
    checks.append(
        ok(name)
        if tuple(weights.shape) == (n, n) and torch.allclose(weights[0], first, atol=1e-6)
        else bad(name, f"got {[round(v, 4) for v in weights.flatten()[:n].tolist()]}\n{_WHERE}")
    )

    # Above the diagonal is the future: those weights must be exactly zero
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    rows_ok = torch.allclose(weights.sum(dim=-1), torch.ones(n), atol=1e-5)
    name = "every weight on a future token is exactly 0, and every row still sums to 1"
    checks.append(
        ok(name)
        if tuple(weights.shape) == (n, n) and bool((weights[future] == 0).all()) and rows_ok
        else bad(name, f"largest future weight {float(weights[future].abs().max()):.4f}, "
                       f"row sums {[round(v, 4) for v in weights.sum(dim=-1).tolist()]}\n{_WHERE}")
    )

    # The same layer weights run through torch's built-in causal attention
    Q, K, V = (F.linear(X, W.T) for W in (layer.W_Q, layer.W_K, layer.W_V))
    reference = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    name = "the output matches torch's F.scaled_dot_product_attention(Q, K, V, is_causal=True)"
    checks.append(
        ok(name)
        if output.shape == reference.shape and torch.allclose(output, reference, atol=1e-6)
        else bad(name, f"expected row 1 = {[round(v, 4) for v in reference[1].tolist()]}\n"
                       f"got      row 1 = {[round(v, 4) for v in output.flatten()[4:8].tolist()]}\n{_WHERE}")
    )
    return checks
