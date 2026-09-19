"""Step 3: get_batch()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_get_batch(get_batch) -> list[Check]:
    """The targets are the inputs shifted one token to the left."""
    checks = []

    # With data = 0, 1, ..., 99 every token equals its position, so the
    # token after x is always x + 1
    data = torch.arange(100)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 4, 3, gen)

    checks.append(
        ok("x and y both have shape (batch_size, block_size): (3, 4) here")
        if tuple(x.shape) == (3, 4) and tuple(y.shape) == (3, 4)
        else bad("x and y both have shape (batch_size, block_size): (3, 4) here",
                 f"x has shape {tuple(x.shape)}, y has shape {tuple(y.shape)} "
                 "(each slice of y should be block_size tokens long)")
    )
    shifted = tuple(y.shape) == tuple(x.shape) and torch.equal(y, x + 1)
    checks.append(
        ok("on data = 0, 1, ..., 99 every target is its input plus one (y = x + 1)")
        if shifted
        else bad("on data = 0, 1, ..., 99 every target is its input plus one (y = x + 1)",
                 f"x[0] = {x[0].tolist()}\ngot y[0] = {y[0].tolist()}, expected {(x[0] + 1).tolist()}")
    )

    # The shortest stream that fits one example: 5 tokens, block_size 4
    x, y = get_batch(torch.arange(5), 4, 1, torch.Generator().manual_seed(0))
    checks.append(
        ok("shortest stream: 0..4 with block_size 4 gives x = [0,1,2,3], y = [1,2,3,4]")
        if x.tolist() == [[0, 1, 2, 3]] and y.tolist() == [[1, 2, 3, 4]]
        else bad("shortest stream: 0..4 with block_size 4 gives x = [0,1,2,3], y = [1,2,3,4]",
                 f"got x = {x.tolist()}, y = {y.tolist()}")
    )
    return checks
