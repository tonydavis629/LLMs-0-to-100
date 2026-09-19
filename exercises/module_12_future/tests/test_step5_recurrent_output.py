"""Step 5: recurrent_step_output()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _show(t):
    """A tensor as a plain list, rounded to 4 decimals for printing."""
    return t.detach().double().round(decimals=4).tolist()


def check_recurrent_step_output(recurrent_step_output) -> list[Check]:
    """(q . S) / (q . z): read one token's output out of the running state."""
    checks = []

    # By hand: q @ S = [1*1 + 2*3, 1*0 + 2*1, 1*2 + 2*0] = [7, 2, 2]
    #          q @ z = 1*2 + 2*1 = 4, so the output is [7, 2, 2] / 4
    q = torch.tensor([1.0, 2.0])
    S = torch.tensor([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0]])
    z = torch.tensor([2.0, 1.0])
    expected = torch.tensor([1.75, 0.5, 0.5])
    out = recurrent_step_output(q, S, z)
    name = "q=[1,2], S=[[1,0,2],[3,1,0]], z=[2,1] gives [7,2,2] / 4 = [1.75, 0.5, 0.5]"
    checks.append(
        ok(name)
        if out.shape == expected.shape and torch.allclose(out, expected)
        else bad(name, f"expected {_show(expected)}\ngot      {_show(out.flatten())}")
    )

    # After a single token, the state holds only that token, so a weighted
    # average over one value must return exactly that value
    k, v = torch.tensor([0.5, 2.0]), torch.tensor([3.0, -1.0, 4.0])
    out = recurrent_step_output(torch.tensor([1.0, 3.0]), torch.outer(k, v), k)
    name = "after one token the output is that token's value: v=[3, -1, 4] comes back out"
    checks.append(
        ok(name)
        if out.shape == v.shape and torch.allclose(out, v, atol=1e-5)
        else bad(name, f"got {_show(out.flatten())} (did you divide by q_t_phi @ z?)")
    )

    # Token by token over a random sequence, against the parallel (masked) form.
    # Positive features stand in for feature_map() output.
    torch.manual_seed(0)
    q_phi, k_phi, V = torch.rand(6, 4) + 0.1, torch.rand(6, 4) + 0.1, torch.randn(6, 3)
    scores = torch.tril(q_phi @ k_phi.T)
    parallel = (scores @ V) / scores.sum(dim=-1, keepdim=True)
    S, z = torch.zeros(4, 3), torch.zeros(4)
    rows = []
    for t in range(6):
        S, z = S + torch.outer(k_phi[t], V[t]), z + k_phi[t]  # a known-good Step 4
        rows.append(recurrent_step_output(q_phi[t], S, z))
    recurrent = torch.stack(rows)
    name = "matches the parallel form row by row on a random 6-token sequence"
    if recurrent.shape != parallel.shape:
        checks.append(bad(name, f"expected shape {tuple(parallel.shape)}, got {tuple(recurrent.shape)}"))
    elif torch.allclose(recurrent, parallel, atol=1e-5):
        checks.append(ok(name))
    else:
        gap = float((recurrent - parallel).abs().max())
        checks.append(bad(name, f"largest gap from the parallel form is {gap:.4f}"))
    return checks
