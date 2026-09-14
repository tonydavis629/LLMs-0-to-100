"""Step 5: the single neuron trained end to end (Steps 1-4 together)

Run by src/main.py. You do NOT need to edit this file. There is no new code
for this step: the runner trains your neuron on both datasets and these
checks confirm it learns the linear one and cannot learn the XOR one.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def check_single_neuron(results: dict) -> list[Check]:
    """`results` holds the losses and (correct, total) from both training runs."""
    checks = []
    lin_losses, lin_correct, lin_total = results["lin_losses"], *results["lin_acc"]
    nl_losses, nl_correct, nl_total = results["nl_losses"], *results["nl_acc"]

    # A line separates the linear data, so a linear classifier should nail it
    lin_pct = 100 * lin_correct / lin_total
    checks.append(
        ok("linear data: accuracy is at least 95%")
        if lin_pct >= 95
        else bad("linear data: accuracy is at least 95%", f"got {lin_correct}/{lin_total} ({lin_pct:.1f}%)")
    )

    # Gradient descent should drive the loss down, a lot
    checks.append(
        ok("linear data: the loss falls to under a quarter of its starting value")
        if lin_losses[-1] < 0.25 * lin_losses[0]
        else bad("linear data: the loss falls to under a quarter of its starting value",
                 f"started at {lin_losses[0]:.4f}, ended at {lin_losses[-1]:.4f}")
    )

    # No line separates XOR: the best a single neuron can do is predict 0.5
    # everywhere, and the cross-entropy of a fair coin is ln 2
    checks.append(
        ok("XOR data: the loss is stuck at ln 2 = 0.693, the cost of a coin flip")
        if abs(nl_losses[-1] - math.log(2)) < 0.01
        else bad("XOR data: the loss is stuck at ln 2 = 0.693, the cost of a coin flip",
                 f"final loss {nl_losses[-1]:.4f}")
    )
    nl_pct = 100 * nl_correct / nl_total
    checks.append(
        ok("XOR data: accuracy is no better than chance (a single neuron is a linear classifier)")
        if nl_pct <= 60
        else bad("XOR data: accuracy is no better than chance (a single neuron is a linear classifier)",
                 f"got {nl_correct}/{nl_total} ({nl_pct:.1f}%)")
    )
    return checks
