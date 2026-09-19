"""Step 1: EmbeddingLayer.forward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _tiny_layer(EmbeddingLayer):
    """A 4-token, 2-dimensional embedding layer with hand-picked tables.

    Token t has the vector [2t-1, 2t] (token 0 is [0, 0]); position p has
    the vector [100(p+1), 100(p+1)], so every sum is easy to read off.
    """
    layer = EmbeddingLayer(vocab_size=4, d_model=2, max_pos=3)
    with torch.no_grad():
        layer.token_embed.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        layer.pos_embed.weight.copy_(torch.tensor([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]))
    return layer


def check_embedding(EmbeddingLayer) -> list[Check]:
    """Token vector plus position vector, for every token in the batch."""
    checks = []
    layer = _tiny_layer(EmbeddingLayer)

    # A batch of 2 sequences, 3 tokens each
    ids = torch.tensor([[2, 1, 2], [0, 0, 0]])
    out = layer(ids)

    # One d_model-sized vector per token
    checks.append(
        ok("returns one vector per token: ids (2, 3) give shape (2, 3, d_model=2)")
        if tuple(out.shape) == (2, 3, 2)
        else bad("returns one vector per token: ids (2, 3) give shape (2, 3, d_model=2)",
                 f"expected shape (2, 3, 2), got {tuple(out.shape)}")
    )
    if tuple(out.shape) != (2, 3, 2):
        return checks

    # Token 2 is [3, 4] and token 1 is [1, 2]; positions 0, 1, 2 add 100, 200, 300
    expected = torch.tensor([[103.0, 104.0], [201.0, 202.0], [303.0, 304.0]])
    name = "adds token + position vectors: ids [2,1,2] give [[103,104],[201,202],[303,304]]"
    if torch.allclose(out[0], expected):
        checks.append(ok(name))
    else:
        got = out[0].detach()
        nudge = ""
        if torch.allclose(got, layer.token_embed.weight[[2, 1, 2]]):
            nudge = "\n(these are the token vectors alone: the position vectors are missing)"
        checks.append(bad(name, f"expected {expected.tolist()}\ngot      {got.tolist()}{nudge}"))

    # Token 0 is the zero vector, so the second row shows the position vectors alone
    expected = torch.tensor([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
    checks.append(
        ok("every sequence in the batch uses positions 0, 1, 2 (row 2: ids [0,0,0])")
        if torch.allclose(out[1], expected)
        else bad("every sequence in the batch uses positions 0, 1, 2 (row 2: ids [0,0,0])",
                 f"expected {expected.tolist()}\ngot      {out[1].detach().tolist()}")
    )
    return checks
