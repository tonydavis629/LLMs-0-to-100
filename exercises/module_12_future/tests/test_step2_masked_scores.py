"""Step 2: masked_scores()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _show(t):
    """A tensor as a plain list, rounded to 4 decimals for printing."""
    return t.detach().double().round(decimals=4).tolist()


def check_masked_scores(masked_scores) -> list[Check]:
    """Every query dotted with every key, with the future multiplied by zero."""
    checks = []

    # By hand: entry (i, j) is q_phi[i] . k_phi[j].
    #   row 0: [1,2].[1,1] = 3   (the future entry [1,2].[2,0] = 2 is zeroed)
    #   row 1: [3,4].[1,1] = 7,  [3,4].[2,0] = 6
    q_phi = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    k_phi = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    expected = torch.tensor([[3.0, 0.0], [7.0, 6.0]])
    out = masked_scores(q_phi, k_phi)
    name = "2 tokens by hand: q_phi=[[1,2],[3,4]], k_phi=[[1,1],[2,0]] gives [[3,0],[7,6]]"
    checks.append(
        ok(name)
        if out.shape == expected.shape and torch.allclose(out, expected)
        else bad(name, f"expected {_show(expected)}\ngot      {_show(out)}")
    )

    # Positive features, like the ones feature_map() produces, for 6 tokens
    torch.manual_seed(0)
    q_rand = torch.rand(6, 3) + 0.1
    k_rand = torch.rand(6, 3) + 0.1
    out_rand = masked_scores(q_rand, k_rand)

    # Everything above the diagonal is a query looking at a later key
    future = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    shape_ok = tuple(out_rand.shape) == (6, 6)
    name = "a (6, 6) matrix with zeros above the diagonal: no token sees the future"
    if not shape_ok:
        checks.append(bad(name, f"expected shape (6, 6), got {tuple(out_rand.shape)}"))
    elif bool((out_rand[future] == 0).all()):
        checks.append(ok(name))
    else:
        largest = float(out_rand[future].abs().max())
        checks.append(bad(name, f"the largest entry above the diagonal is {largest:.4f} "
                                "(did you multiply by mask?)"))

    # The same computation written with torch built-ins
    reference = torch.tril(q_rand @ k_rand.T)
    name = "matches torch.tril(q_phi @ k_phi.T) on a random 6-token input"
    if shape_ok and torch.allclose(out_rand, reference, atol=1e-6):
        checks.append(ok(name))
    elif shape_ok and torch.allclose(out_rand, torch.tril(k_rand @ q_rand.T), atol=1e-6):
        checks.append(bad(name, "you scored keys against queries: row i should be query i "
                                "(put q_phi first and transpose k_phi)"))
    else:
        checks.append(bad(name, "row i should hold query i dotted with keys 0 to i, "
                                "and zeros after that"))
    return checks
