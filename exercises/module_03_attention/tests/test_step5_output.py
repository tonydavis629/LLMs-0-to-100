"""Step 5: TinyAttentionLayer.attention_output()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_attention_output(TinyAttentionLayer) -> list[Check]:
    """Each query's output is the weighted average of the value rows."""
    checks = []
    layer = TinyAttentionLayer(d_model=4, d_k=4)

    # Query 0 puts all its weight on value 0; query 1 splits 50/50
    weights = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    V = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    out = layer.attention_output(weights, V)
    expected = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    name = "weights [[1,0],[0.5,0.5]] over values [[2,0],[0,4]] give [[2,0],[1,2]]"
    if out.shape == expected.shape and torch.allclose(out, expected, atol=1e-6):
        checks.append(ok(name))
    else:
        nudge = ""
        if out.shape == expected.shape and not torch.allclose(out, expected):
            nudge = " (the weights go on the left: row i of weights mixes the rows of V)"
        checks.append(bad(name, f"expected {expected.tolist()}\ngot      {out.tolist()}{nudge}"))

    # 3 queries attending over 2 values of width 4: one output row per query
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(3, 2), dim=-1)
    out = layer.attention_output(weights, torch.randn(2, 4))
    name = "one output row per query: 3 queries over 2 values of width 4 give shape (3, 4)"
    checks.append(
        ok(name)
        if tuple(out.shape) == (3, 4)
        else bad(name, f"expected (3, 4), got {tuple(out.shape)}")
    )

    # Softmax weights from random Q and K, then your weighted sum, against torch's attention
    Q, K, V = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    weights = torch.softmax(Q @ K.T / 2.0, dim=-1)
    out = layer.attention_output(weights, V)
    reference = F.scaled_dot_product_attention(Q, K, V)
    name = "with softmax weights, matches torch's F.scaled_dot_product_attention (5 tokens, d_k=4)"
    checks.append(
        ok(name)
        if out.shape == reference.shape and torch.allclose(out, reference, atol=1e-5)
        else bad(name, f"expected row 0 = {[round(v, 4) for v in reference[0].tolist()]}\n"
                       f"got      row 0 = {[round(v, 4) for v in out.flatten()[:4].tolist()]}")
    )
    return checks
