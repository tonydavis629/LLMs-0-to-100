"""Learning-rate schedules provided for you.

You do NOT need to edit this file. The training loop calls `lr_at_step()`
once per step: linear warmup, then cosine decay down to a floor.
"""

from __future__ import annotations

import math


def lr_at_step(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Return the learning rate for a given step: warmup then cosine decay.

    Args:
        step: The current training step (0-indexed).
        warmup_steps: Steps spent linearly ramping the LR up to max_lr.
        max_steps: Total steps; LR reaches min_lr at the end.
        max_lr: Peak learning rate (reached at the end of warmup).
        min_lr: Final/minimum learning rate.

    Returns:
        The learning rate to use at this step.
    """
    # 1) Warmup: ramp linearly from ~0 up to max_lr.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) After the schedule ends, hold at the floor.
    if step > max_steps:
        return min_lr
    # 3) Cosine decay from max_lr down to min_lr.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
