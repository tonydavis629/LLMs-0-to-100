"""GRPO training (Steps 1-10 together)

Run by src/main.py after the training loop. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def check_training(results: dict) -> list[Check]:
    """With every piece in place, reward should climb and held-out accuracy should rise."""
    checks = []

    # The reward curve: mean group reward at the first and last checkpoints
    rewards = results["curve_rewards"]
    first, last = rewards[0], rewards[-1]
    name = "the mean group reward climbs by at least 0.3 from the first checkpoint to the last"
    checks.append(
        ok(name)
        if last - first >= 0.3
        else bad(name, f"went from {first:.3f} to {last:.3f} "
                       "(a falling curve usually means a sign error in pg_loss() or the advantages)")
    )

    # Held-out prompts the policy never trained on, sampled at temperature 1.0
    before, after = results["acc_before"], results["acc_after"]
    name = "held-out sampled accuracy rises by at least 20 points"
    checks.append(
        ok(name)
        if after - before >= 0.20
        else bad(name, f"went from {before:.1%} to {after:.1%}")
    )

    # The argmax answer should not get worse while sampling gets more reliable
    before, after = results["greedy_before"], results["greedy_after"]
    name = "held-out greedy accuracy does not fall"
    checks.append(
        ok(name)
        if after >= before
        else bad(name, f"went from {before:.1%} to {after:.1%}")
    )
    return checks
