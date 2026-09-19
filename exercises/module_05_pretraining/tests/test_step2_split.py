"""Step 2: train_val_split()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _as_list(t) -> list:
    """Tensor or list to a plain list, for printing."""
    return t.tolist() if isinstance(t, torch.Tensor) else list(t)


def check_train_val_split(train_val_split) -> list[Check]:
    """A prefix for training, the tail held out, nothing lost or shuffled."""
    checks = []

    # Ten tokens with val_fraction = 0.2: the first 8 train, the last 2 validate
    train, val = train_val_split(torch.arange(10), 0.2)
    checks.append(
        ok("0..9 with val_fraction=0.2 gives train [0..7] and validation [8, 9]")
        if _as_list(train) == list(range(8)) and _as_list(val) == [8, 9]
        else bad("0..9 with val_fraction=0.2 gives train [0..7] and validation [8, 9]",
                 f"got train {_as_list(train)}, validation {_as_list(val)} "
                 "(return (train, val) in that order, with val the LAST fraction)")
    )

    # The default holds out 10%, and the two pieces together are the whole stream
    data = torch.arange(100)
    train, val = train_val_split(data)
    sizes_ok = len(train) == 90 and len(val) == 10
    rejoined = torch.cat([torch.as_tensor(train), torch.as_tensor(val)])
    checks.append(
        ok("the default val_fraction=0.1 splits 100 tokens into 90 and 10")
        if sizes_ok
        else bad("the default val_fraction=0.1 splits 100 tokens into 90 and 10",
                 f"got {len(train)} train and {len(val)} validation tokens")
    )
    checks.append(
        ok("train + validation is the original stream: nothing lost, repeated, or shuffled")
        if torch.equal(rejoined, data)
        else bad("train + validation is the original stream: nothing lost, repeated, or shuffled",
                 f"train + validation has {len(rejoined)} tokens starting "
                 f"{rejoined[:5].tolist()} and ending {rejoined[-5:].tolist()}")
    )
    return checks
