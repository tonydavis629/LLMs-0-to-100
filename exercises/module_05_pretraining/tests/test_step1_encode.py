"""Step 1: encode()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_encode(encode, text: str, stoi: dict[str, int]) -> list[Check]:
    """Characters in, integer IDs out: a hand example, the dtype, and a round trip."""
    checks = []

    # A three-letter vocabulary, so the answer can be read off by eye
    ids = encode("abca", {"a": 0, "b": 1, "c": 2})
    got = ids.tolist() if isinstance(ids, torch.Tensor) else ids
    checks.append(
        ok("looks up every character: 'abca' with a=0, b=1, c=2 gives [0, 1, 2, 0]")
        if got == [0, 1, 2, 0]
        else bad("looks up every character: 'abca' with a=0, b=1, c=2 gives [0, 1, 2, 0]",
                 f"got {got}")
    )

    # nn.Embedding and F.cross_entropy both need a 1-D tensor of int64 IDs
    is_long = isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.dim() == 1
    checks.append(
        ok("returns a 1-D torch.long tensor (the dtype nn.Embedding needs)")
        if is_long
        else bad("returns a 1-D torch.long tensor (the dtype nn.Embedding needs)",
                 f"got {type(ids).__name__} with dtype {getattr(ids, 'dtype', None)} "
                 f"and shape {tuple(getattr(ids, 'shape', ()))} (did you pass dtype=torch.long?)")
    )

    # Decoding the IDs must give back exactly the text we started from
    sample = text[:1000]
    itos = {i: c for c, i in stoi.items()}
    back = "".join(itos.get(int(i), "?") for i in encode(sample, stoi))
    checks.append(
        ok("round trip: decoding the IDs of the first 1,000 characters gives the text back")
        if back == sample
        else bad("round trip: decoding the IDs of the first 1,000 characters gives the text back",
                 f"expected {sample[:40]!r}...\ngot      {back[:40]!r}...")
    )
    return checks
