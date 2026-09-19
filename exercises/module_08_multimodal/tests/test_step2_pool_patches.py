"""Step 2: pool_patches()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_pool_patches(pool_patches) -> list[Check]:
    """One vector per image: the average of that image's patch vectors."""
    checks = []

    # One image, two patches of width 3: the average is taken position by position
    got = pool_patches(torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]))
    name = "averages the patches: [1,2,3] and [3,4,5] pool to [2, 3, 4]"
    if tuple(got.shape) == (1, 3) and torch.allclose(got, torch.tensor([[2.0, 3.0, 4.0]])):
        checks.append(ok(name))
    elif tuple(got.shape) == (1, 3) and torch.allclose(got, torch.tensor([[4.0, 6.0, 8.0]])):
        checks.append(bad(name, "expected [[2.0, 3.0, 4.0]], got [[4.0, 6.0, 8.0]] (did you sum instead of average?)"))
    else:
        checks.append(bad(name, f"expected [[2.0, 3.0, 4.0]] with shape (1, 3), "
                                f"got {got.tolist()} with shape {tuple(got.shape)}"))

    # The real sizes: 16 patches of width 64 collapse to one 64-wide vector per image
    got = pool_patches(torch.zeros(2, 16, 64))
    checks.append(
        ok("returns one vector per image: (B, N, D) = (2, 16, 64) gives (2, 64)")
        if tuple(got.shape) == (2, 64)
        else bad("returns one vector per image: (B, N, D) = (2, 16, 64) gives (2, 64)",
                 f"got shape {tuple(got.shape)} (average over dim=1, the patch dimension)")
    )

    # Two images with two patches each. Averaging over the batch (dim=0) gives the
    # same (2, 2) shape here, so only the values reveal that mistake.
    x = torch.tensor([[[0.0, 0.0], [2.0, 2.0]],       # image 0
                      [[10.0, 10.0], [20.0, 20.0]]])  # image 1
    got = pool_patches(x)
    expected = torch.tensor([[1.0, 1.0], [15.0, 15.0]])
    name = "pools each image separately: patches [0,0],[2,2] -> [1,1] and [10,10],[20,20] -> [15,15]"
    checks.append(
        ok(name)
        if got.shape == expected.shape and torch.allclose(got, expected)
        else bad(name, f"expected {expected.tolist()}, got {got.tolist()} "
                       "(average over the patches of one image, not across images)")
    )
    return checks
