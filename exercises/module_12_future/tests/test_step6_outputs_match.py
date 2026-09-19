"""Step 6: outputs_match()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_outputs_match(outputs_match) -> list[Check]:
    """True when every pair of entries is within the tolerance, False otherwise."""
    checks = []
    zeros = torch.zeros(2)

    # The easy case: a tensor always matches itself
    a = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    checks.append(
        ok("a tensor matches itself")
        if bool(outputs_match(a, a.clone()))
        else bad("a tensor matches itself", "expected True, got False")
    )

    # Floating-point noise must not count as a difference. An exact == check
    # fails here, and so does torch.allclose with its default tolerance.
    close = torch.tensor([0.0, 0.00005])
    checks.append(
        ok("a gap below the tolerance still matches: [0, 0] vs [0, 0.00005] gives True")
        if bool(outputs_match(zeros, close))
        else bad("a gap below the tolerance still matches: [0, 0] vs [0, 0.00005] gives True",
                 "expected True, got False (did you pass atol=tolerance?)")
    )

    # A real difference must be caught, whichever tensor is larger
    far = torch.tensor([0.0, 0.01])
    checks.append(
        ok("a gap above the tolerance does not: [0, 0] vs [0, 0.01] gives False")
        if not bool(outputs_match(zeros, far)) and not bool(outputs_match(far, zeros))
        else bad("a gap above the tolerance does not: [0, 0] vs [0, 0.01] gives False",
                 "expected False, got True (compare the absolute difference to the tolerance)")
    )

    # The caller's tolerance must be the one that is used
    loose = torch.tensor([0.0, 0.05])
    checks.append(
        ok("uses the tolerance it is given: a 0.05 gap matches when tolerance=0.1")
        if bool(outputs_match(zeros, loose, tolerance=0.1))
        else bad("uses the tolerance it is given: a 0.05 gap matches when tolerance=0.1",
                 "expected True, got False (did you use the tolerance argument?)")
    )
    return checks
