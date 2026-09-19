"""Step 5: completion_mask()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_completion_mask(completion_mask) -> list[Check]:
    """True exactly at the positions whose next token was generated."""
    checks = []

    # 3 prompt tokens + 3 generated tokens: positions 0..4, and position t predicts token t + 1.
    # Tokens 3, 4, 5 are generated, so positions 2, 3, 4 are the completion positions.
    mask = completion_mask(3, 6)
    expected = [False, False, True, True, True]
    name = "prompt_len=3, seq_len=6 gives [F, F, T, T, T]: position t predicts token t + 1"
    got = mask.tolist() if isinstance(mask, torch.Tensor) else mask
    if got == expected:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected {expected}, got {got} "
                                "(the first generated token is predicted at position prompt_len - 1)"))

    mask = completion_mask(4, 10)
    name = "returns a bool tensor of length seq_len - 1: prompt_len=4, seq_len=10 gives length 9"
    if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and tuple(mask.shape) == (9,):
        checks.append(ok(name))
    else:
        kind = f"dtype {mask.dtype}, shape {tuple(mask.shape)}" if isinstance(mask, torch.Tensor) \
            else type(mask).__name__
        checks.append(bad(name, f"expected dtype torch.bool, shape (9,); got {kind}"))

    # One True per generated token, whatever the lengths
    mask = completion_mask(17, 23)
    count = int(torch.as_tensor(mask).sum())
    name = "one True per generated token: prompt_len=17, seq_len=23 has 6 True positions"
    if count == 6:
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected 6 True positions, got {count}"))
    return checks
