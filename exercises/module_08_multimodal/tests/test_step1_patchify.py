"""Step 1: patchify()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_patchify(patchify) -> list[Check]:
    """Cut images into P x P squares, laid out left to right, then top to bottom."""
    checks = []

    # A 4 x 4 one-channel image numbered 0..15, cut into 2 x 2 patches:
    #    0  1 |  2  3
    #    4  5 |  6  7
    #   ------+------
    #    8  9 | 10 11
    #   12 13 | 14 15
    image = torch.arange(16.0).reshape(1, 1, 4, 4)
    got = patchify(image, 2)
    expected = torch.tensor([
        [[0.0, 1.0], [4.0, 5.0]],      # top-left
        [[2.0, 3.0], [6.0, 7.0]],      # top-right
        [[8.0, 9.0], [12.0, 13.0]],    # bottom-left
        [[10.0, 11.0], [14.0, 15.0]],  # bottom-right
    ]).reshape(1, 4, 1, 2, 2)
    name = "patches come in row-major order: 4 x 4 image 0-15 gives [[0,1],[4,5]], [[2,3],[6,7]], ..."
    if tuple(got.shape) != (1, 4, 1, 2, 2):
        checks.append(bad(name, f"expected shape (1, 4, 1, 2, 2), got {tuple(got.shape)}"))
    elif torch.equal(got, expected):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected the first two patches {expected[0, :2, 0].long().tolist()}\n"
                                f"got {got[0, :2, 0].long().tolist()}\n"
                                "(did you permute the two grid axes next to the batch before the last reshape?)"))

    # Shape: 2 images, 3 channels, 8 x 8 pixels, P = 4 -> a 2 x 2 grid of 4 patches each
    torch.manual_seed(0)
    got = patchify(torch.rand(2, 3, 8, 8), 4)
    checks.append(
        ok("returns (B, N, C, P, P): a 2 x 3 x 8 x 8 batch with P=4 gives (2, 4, 3, 4, 4)")
        if tuple(got.shape) == (2, 4, 3, 4, 4)
        else bad("returns (B, N, C, P, P): a 2 x 3 x 8 x 8 batch with P=4 gives (2, 4, 3, 4, 4)",
                 f"got shape {tuple(got.shape)}")
    )

    # Put the patches back with F.fold (the inverse of cutting them out). A correct
    # patchify loses nothing, duplicates nothing, and keeps each channel separate.
    # The image is not square (4 x 6), so mixing up height and width shows up here.
    images = torch.rand(2, 3, 4, 6)
    got = patchify(images, 2)
    name = "un-patchifying a random 2 x 3 x 4 x 6 batch gives back the original images"
    if tuple(got.shape) != (2, 6, 3, 2, 2):
        checks.append(bad(name, f"expected shape (2, 6, 3, 2, 2), got {tuple(got.shape)}"))
    else:
        columns = got.reshape(2, 6, 3 * 2 * 2).transpose(1, 2)  # (B, C*P*P, N), what F.fold expects
        rebuilt = F.fold(columns, output_size=(4, 6), kernel_size=2, stride=2)
        checks.append(
            ok(name)
            if torch.allclose(rebuilt, images)
            else bad(name, "the patches do not reassemble into the input images\n"
                           "(check the permute: the grid rows H/P come before the grid columns W/P)")
        )
    return checks
