"""Step 6: image_to_prefix()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


def check_image_to_prefix(image_to_prefix) -> list[Check]:
    """Apply the projector, then cut its output into prefix_len vectors of width d_llm."""
    checks = []

    # 2 image embeddings of width 8, K = 4 prefix vectors of width 16 (so 64 outputs)
    torch.manual_seed(0)
    to_prefix = nn.Linear(8, 4 * 16)
    x = torch.randn(2, 8)
    with torch.no_grad():
        got = image_to_prefix(x, to_prefix, 4)
    name = "returns (B, prefix_len, d_llm): 2 embeddings, K=4, Linear(8 -> 64) give (2, 4, 16)"
    checks.append(
        ok(name)
        if tuple(got.shape) == (2, 4, 16)
        else bad(name, f"got shape {tuple(got.shape)}")
    )

    # Reference: the same projector applied by hand, weight and bias included
    with torch.no_grad():
        expected = (x @ to_prefix.weight.t() + to_prefix.bias).reshape(2, 4, 16)
    name = "applies the projector, weights and bias: matches the Linear computed by hand"
    checks.append(
        ok(name)
        if got.shape == expected.shape and torch.allclose(got, expected, atol=1e-5)
        else bad(name, "the values differ from to_prefix(image_embeds) reshaped to (B, K, d_llm)\n"
                       "(did you apply to_prefix before reshaping?)")
    )

    # Order: an identity projector passes [0, 1, ..., 11] through unchanged, and
    # K = 3 prefix vectors of width 4 are consecutive slices of it
    identity = nn.Linear(12, 12)
    with torch.no_grad():
        identity.weight.copy_(torch.eye(12))
        identity.bias.zero_()
        got = image_to_prefix(torch.arange(12.0).reshape(1, 12), identity, 3)
    expected = torch.arange(12.0).reshape(1, 3, 4)
    name = "prefix vector k is slice k of the output: [0..11], K=3 gives [0..3], [4..7], [8..11]"
    checks.append(
        ok(name)
        if got.shape == expected.shape and torch.equal(got, expected)
        else bad(name, f"expected {expected[0].tolist()}\ngot {got.tolist()}")
    )
    return checks
