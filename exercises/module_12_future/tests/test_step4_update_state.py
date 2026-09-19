"""Step 4: update_state()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def _show(t):
    """A tensor as a plain list, rounded to 4 decimals for printing."""
    return t.detach().double().round(decimals=4).tolist()


def check_update_state(update_state) -> list[Check]:
    """S gains the outer product of key and value; z gains the key."""
    checks = []

    # By hand, from an empty state: outer([1, 2], [3, 4]) = [[3, 4], [6, 8]]
    S, z = update_state(torch.zeros(2, 2), torch.zeros(2),
                        torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    exp_S, exp_z = torch.tensor([[3.0, 4.0], [6.0, 8.0]]), torch.tensor([1.0, 2.0])
    name = "first token from an empty state: k=[1,2], v=[3,4] gives S=[[3,4],[6,8]], z=[1,2]"
    checks.append(
        ok(name)
        if S.shape == exp_S.shape and torch.allclose(S, exp_S) and torch.allclose(z, exp_z)
        else bad(name, f"expected S={_show(exp_S)}, z={_show(exp_z)}\n"
                       f"got      S={_show(S)}, z={_show(z)}")
    )

    # A state that already holds something (all ones), with d=2 keys and d_v=3 values
    exp_S = torch.tensor([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])
    exp_z = torch.tensor([2.0, 1.0])
    name = "adds to the state rather than replacing it: S and z start as all ones"
    try:
        S, z = update_state(torch.ones(2, 3), torch.ones(2),
                            torch.tensor([1.0, 0.0]), torch.tensor([1.0, 2.0, 3.0]))
    except RuntimeError as e:
        # Usually a shape mismatch that only shows up when d and d_v differ
        checks.append(bad(name, f"crashed with d=2 and d_v=3: {e}\n"
                                "(S is (d, d_v), so its update is torch.outer(k_t_phi, v_t))"))
        return checks
    checks.append(
        ok(name)
        if S.shape == exp_S.shape and torch.allclose(S, exp_S) and torch.allclose(z, exp_z)
        else bad(name, f"expected S={_show(exp_S)}, z={_show(exp_z)}\n"
                       f"got      S={_show(S)}, z={_show(z)}")
    )

    # Absorbing every token one at a time should add up to the same sums the
    # parallel form uses: S = K^T V and z = the sum of the keys
    torch.manual_seed(0)
    K, V = torch.rand(10, 4), torch.randn(10, 3)
    S, z = torch.zeros(4, 3), torch.zeros(4)
    for t in range(10):
        S, z = update_state(S, z, K[t], V[t])
    name = "10 tokens one at a time give S = K^T V and z = the sum of the keys"
    if S.shape != (4, 3) or z.shape != (4,):
        checks.append(bad(name, f"expected shapes (4, 3) and (4,), "
                                f"got {tuple(S.shape)} and {tuple(z.shape)}"))
    elif torch.allclose(S, K.T @ V, atol=1e-5) and torch.allclose(z, K.sum(0), atol=1e-5):
        checks.append(ok(name))
    else:
        checks.append(bad(name, "the state does not hold the sum over all 10 tokens "
                                "(each update should add to S and z, not replace them)"))
    return checks
