"""Step 4: similarity_matrix()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_similarity_matrix(similarity_matrix) -> list[Check]:
    """Entry (i, j) is image i dotted with caption j, divided by the temperature."""
    checks = []

    # Image 0 points along x, image 1 along y; both captions point along x.
    # So image 0 matches both captions (row [1, 1]) and image 1 matches neither.
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    captions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    got = similarity_matrix(images, captions, 1.0)
    expected = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    name = "rows are images: images [[1,0],[0,1]] x captions [[1,0],[1,0]] give [[1,1],[0,0]]"
    if got.shape == expected.shape and torch.allclose(got, expected):
        checks.append(ok(name))
    elif got.shape == expected.shape and torch.allclose(got, expected.t()):
        checks.append(bad(name, f"expected {expected.tolist()}, got {got.tolist()} "
                                "(the matrix is transposed: put image_embeds first)"))
    else:
        checks.append(bad(name, f"expected {expected.tolist()}, got {got.tolist()}"))

    # A temperature of 0.5 should double every similarity
    got = similarity_matrix(images, captions, 0.5)
    name = "divides by the temperature: temperature 0.5 doubles every entry to [[2,2],[0,0]]"
    if got.shape == expected.shape and torch.allclose(got, 2 * expected):
        checks.append(ok(name))
    elif got.shape == expected.shape and torch.allclose(got, 0.5 * expected):
        checks.append(bad(name, f"expected {(2 * expected).tolist()}, got {got.tolist()} "
                                "(divide by the temperature, do not multiply)"))
    else:
        checks.append(bad(name, f"expected {(2 * expected).tolist()}, got {got.tolist()}"))

    # For unit vectors, every entry is a cosine similarity (scaled by 1/temperature)
    torch.manual_seed(0)
    img = F.normalize(torch.randn(3, 4), dim=-1)
    txt = F.normalize(torch.randn(3, 4), dim=-1)
    got = similarity_matrix(img, txt, 0.07)
    reference = F.cosine_similarity(img.unsqueeze(1), txt.unsqueeze(0), dim=-1) / 0.07
    name = "entry (i, j) is cosine(image i, caption j) / T on random unit vectors (T=0.07)"
    checks.append(
        ok(name)
        if got.shape == reference.shape and torch.allclose(got, reference, atol=1e-4)
        else bad(name, f"expected row 0 = {[round(v, 3) for v in reference[0].tolist()]}, shape (3, 3)\n"
                       f"got first values {[round(v, 3) for v in got.flatten()[:3].tolist()]}, "
                       f"shape {tuple(got.shape)}")
    )
    return checks
