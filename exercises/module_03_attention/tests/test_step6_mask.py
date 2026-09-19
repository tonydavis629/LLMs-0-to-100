"""Step 6: TinyAttentionLayer.causal_mask()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok

INF = float("inf")


def check_causal_mask(TinyAttentionLayer) -> list[Check]:
    """0 where token i may look at token j (j <= i), -inf where j is in the future."""
    checks = []
    layer = TinyAttentionLayer(d_model=4, d_k=4)

    # The whole 3 x 3 mask, written out
    mask = layer.causal_mask(3)
    expected = torch.tensor([[0.0, -INF, -INF], [0.0, 0.0, -INF], [0.0, 0.0, 0.0]])
    name = "seq_len=3 gives [[0,-inf,-inf], [0,0,-inf], [0,0,0]]"
    if mask.shape == expected.shape and torch.equal(mask.float(), expected):
        checks.append(ok(name))
    else:
        nudge = ""
        if bool((mask == 1).any()):
            nudge = "\n(the allowed 1s still need to become 0.0)"
        elif bool(torch.isinf(torch.tril(mask.float())).any()):
            nudge = "\n(-inf belongs only above the diagonal, where j > i; token i still sees tokens 0..i)"
        elif not bool(torch.isinf(mask).any()):
            nudge = "\n(blocked entries should be float('-inf'))"
        checks.append(bad(name, f"expected {expected.tolist()}\ngot      {mask.tolist()}{nudge}"))

    # Add the mask to 4 equal scores: row i should spread evenly over tokens 0..i
    weights = torch.softmax(torch.zeros(4, 4) + layer.causal_mask(4), dim=-1)
    expected = torch.tensor([[1 / (i + 1) if j <= i else 0.0 for j in range(4)] for i in range(4)])
    name = "after softmax, 4 equal scores give row i weight 1/(i+1) on tokens 0..i and exactly 0 after"
    checks.append(
        ok(name)
        if weights.shape == expected.shape and torch.allclose(weights, expected, atol=1e-6)
        else bad(name, f"expected row 1 = [0.5, 0.5, 0.0, 0.0], got {[round(v, 4) for v in weights[1].tolist()]}")
    )

    # Your mask handed to torch's attention should match torch's built-in causal option
    torch.manual_seed(0)
    Q, K, V = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    yours = F.scaled_dot_product_attention(Q, K, V, attn_mask=layer.causal_mask(5).float())
    reference = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    name = "used as attn_mask on 5 tokens, matches torch's is_causal=True attention"
    checks.append(
        ok(name)
        if torch.allclose(yours, reference, atol=1e-5)
        else bad(name, f"expected row 1 = {[round(v, 4) for v in reference[1].tolist()]}\n"
                       f"got      row 1 = {[round(v, 4) for v in yours[1].tolist()]}")
    )
    return checks
