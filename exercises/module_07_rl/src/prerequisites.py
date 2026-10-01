"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Several steps call code from earlier
ones: Step 4 needs a scored group from Steps 1-3, and training needs all
ten. If Step 2 were unfinished, running Step 4 would report Step 2's TODO
under Step 4's header. `require()` avoids that: it first tries the current
step's own blank on a tiny input, then each earlier step it depends on, and
names the ones that are still missing.
"""

from __future__ import annotations

import torch

from exercise import (
    completion_mask,
    gather_token_log_probs,
    grpo_step,
    group_relative_advantages,
    kl_penalty,
    mean_reward,
    pg_loss,
    sample_group,
    score_group,
    verifiable_reward,
)

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    1: "sample_group",
    2: "verifiable_reward",
    3: "score_group",
    4: "group_relative_advantages",
    5: "completion_mask",
    6: "gather_token_log_probs",
    7: "pg_loss",
    8: "kl_penalty",
    9: "grpo_step",
    10: "mean_reward",
}


def _fake_generate(*args, **kwargs) -> torch.Tensor:
    """Stand in for the model's generate(): always return the same two tokens."""
    return torch.zeros(1, 2, dtype=torch.long)


def _probe_grpo_step() -> None:
    """Call grpo_step once on a throwaway one-weight layer."""
    layer = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.0)  # lr 0: nothing actually changes
    grpo_step(optimizer, layer(torch.zeros(1, 1)).sum(), layer, 1.0)


# One call per step on small throwaway inputs; only a NotImplementedError matters
_PROBES = {
    1: lambda: sample_group(None, torch.zeros(1, 1, dtype=torch.long), 1, 1, 128, 1.0, _fake_generate, None),
    2: lambda: verifiable_reward("tac", "tac"),
    3: lambda: score_group(["tac"], "tac"),
    4: lambda: group_relative_advantages(torch.tensor([1.0, 0.0])),
    5: lambda: completion_mask(3, 6),
    6: lambda: gather_token_log_probs(torch.zeros(2, 3), torch.tensor([0, 1])),
    7: lambda: pg_loss(torch.zeros(3), 1.0, torch.ones(3, dtype=torch.bool)),
    8: lambda: kl_penalty(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool)),
    9: _probe_grpo_step,
    10: lambda: mean_reward(torch.tensor([1.0, 0.0])),
}

# score_group() calls verifiable_reward(), so an unfinished Step 2 can raise from
# inside a finished Step 3. That TODO belongs to Step 2, not Step 3.
_CALLS_INTO = {3: [2]}


def _todo_text(step: int) -> str | None:
    """The step's own TODO message if its blank is unfinished, else None."""
    try:
        _PROBES[step]()
    except NotImplementedError as e:
        if any(str(e) == _todo_text(n) for n in _CALLS_INTO.get(step, [])):
            return None  # an earlier step's TODO, raised from inside this one
        return str(e)
    except Exception:  # noqa: BLE001 - it ran, so it is not a TODO (its own tests judge it)
        return None
    return None


def require(step: int | None, needs=(), why: str = "") -> None:
    """Raise NotImplementedError if `step` or any step in `needs` is unfinished.

    The current step is tried first so that its own TODO message is the one
    reported. Missing earlier steps get a "needs Step N" message instead.
    Pass step=None when there is no blank of its own (the training run).
    """
    if step is not None:
        message = _todo_text(step)
        if message is not None:
            raise NotImplementedError(message)

    missing = [n for n in needs if _todo_text(n) is not None]
    if len(missing) == 1:
        raise NotImplementedError(f"needs Step {missing[0]} ({STEP_NAMES[missing[0]]}) {why}")
    if missing:
        numbers = ", ".join(str(n) for n in missing[:-1]) + f" and {missing[-1]}"
        if len(missing) == len(STEP_NAMES):
            numbers = f"1-{len(STEP_NAMES)}"  # all of them
        raise NotImplementedError(f"needs Steps {numbers} {why}")
