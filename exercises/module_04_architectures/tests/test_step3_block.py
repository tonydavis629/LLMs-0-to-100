"""Step 3: TransformerBlock.forward()

Run by src/main.py. You do NOT need to edit this file. The tests replace
the block's feed-forward network with their own, so this step does not
depend on your Step 2.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


class Scale(nn.Module):
    """A stand-in sub-layer that multiplies its input by a fixed number."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor * x


def _reference_ffn(d_model: int, d_ff: int) -> nn.Module:
    """A correct GPT-2 feed-forward network built from torch layers."""
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(approximate="tanh"), nn.Linear(d_ff, d_model))


def check_transformer_block(TransformerBlock) -> list[Check]:
    """Pre-norm residual structure: x + attn(ln1(x)), then x + ffn(ln2(x))."""
    checks = []

    # A real block with a reference FFN and random layer-norm parameters
    torch.manual_seed(0)
    block = TransformerBlock(d_model=8, n_heads=2, d_ff=16, dropout=0.0).eval()
    block.ffn = _reference_ffn(8, 16)
    with torch.no_grad():
        for ln in (block.ln1, block.ln2):
            ln.weight.uniform_(0.5, 1.5)
            ln.bias.uniform_(-0.5, 0.5)
    x = torch.randn(2, 5, 8)
    out = block(x)

    checks.append(
        ok("keeps the shape: (2, 5, 8) in gives (2, 5, 8) out")
        if tuple(out.shape) == (2, 5, 8)
        else bad("keeps the shape: (2, 5, 8) in gives (2, 5, 8) out",
                 f"expected (2, 5, 8), got {tuple(out.shape)}")
    )
    if tuple(out.shape) != (2, 5, 8):
        return checks

    # Hand example: identity norms, attn(x) = 2x and ffn(x) = 10x. Starting from
    # x = 1, attention gives 1 + 2 = 3, then the FFN gives 3 + 30 = 33
    toy = TransformerBlock(d_model=4, n_heads=2, d_ff=8, dropout=0.0).eval()
    toy.ln1, toy.ln2 = nn.Identity(), nn.Identity()
    toy.attn, toy.ffn = Scale(2.0), Scale(10.0)
    got = toy(torch.ones(1, 2, 4))
    name = "adds each sub-layer back to its input: attn(x)=2x, ffn(x)=10x turn x=1 into 33"
    if torch.allclose(got, torch.full((1, 2, 4), 33.0)):
        checks.append(ok(name))
    else:
        value = float(got.flatten()[0])
        nudges = {
            20.0: "\n(20 means no residual at all: add x back after each sub-layer)",
            13.0: "\n(the FFN should read the updated x, after the attention residual)",
            30.0: "\n(the FFN output needs its own residual: x = x + ...)",
            22.0: "\n(the attention output needs its own residual: x = x + ...)",
        }
        checks.append(bad(name, f"expected 33\ngot      {value:g}{nudges.get(round(value, 3), '')}"))

    # Recompute the pre-norm block from its own layers and compare
    with torch.no_grad():
        want = x + block.attn(block.ln1(x))
        want = want + block.ffn(block.ln2(want))
        post = block.ln1(x + block.attn(x))
        post = block.ln2(post + block.ffn(post))
    name = "normalizes before each sub-layer: x + attn(ln1(x)), then x + ffn(ln2(x))"
    if torch.allclose(out.detach(), want, atol=1e-5):
        checks.append(ok(name))
    elif torch.allclose(out.detach(), post, atol=1e-5):
        checks.append(bad(name, "got the post-norm order ln(x + sublayer(x)) of the 2017 Transformer\n"
                                "(GPT-2 applies ln1/ln2 to the input of each sub-layer instead)"))
    else:
        diff = float((out.detach() - want).abs().max())
        checks.append(bad(name, f"largest difference from the reference is {diff:.4f}"))
    return checks
