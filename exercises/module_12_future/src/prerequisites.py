"""Checking which steps are finished, provided for you.

You do NOT need to edit this file. Several steps call the code from earlier
steps. If Step 2 were unfinished, running Step 3 would report Step 2's TODO
under Step 3's header. `require()` avoids that: it first tries the current
step's own blank on a tiny input, then each earlier step it depends on, and
names the first one that is still missing. `is_done()` answers the same
question for one step, so Step 7 can time only the forms that are ready.
"""

from __future__ import annotations

import torch

from exercise import (
    feature_map,
    masked_scores,
    outputs_match,
    parallel_linear_attention,
    recurrent_step_output,
    time_forward,
    update_state,
)
from tests.test_step3_parallel import known_good_steps_1_and_2

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    "1": "feature_map",
    "2": "masked_scores",
    "3": "parallel_linear_attention",
    "4": "update_state",
    "5": "recurrent_step_output",
    "6": "outputs_match",
    "7": "time_forward",
}

# Throwaway inputs for the probes below
_TINY = torch.ones(2, 2)
_VEC = torch.ones(2)


def _probe_parallel():
    """Run Step 3 with known-good Steps 1 and 2 swapped in, so only Step 3's own line runs."""
    with known_good_steps_1_and_2(parallel_linear_attention):
        return parallel_linear_attention(_TINY, _TINY, _TINY)


# One tiny call per step. Each raises NotImplementedError while that step's
# blank is unfinished, and runs in a fraction of a second.
_PROBES = {
    "1": lambda: feature_map(_TINY),
    "2": lambda: masked_scores(_TINY, _TINY),
    "3": _probe_parallel,
    "4": lambda: update_state(_TINY, _VEC, _VEC, _VEC),
    "5": lambda: recurrent_step_output(_VEC, _TINY, _VEC),
    "6": lambda: outputs_match(_TINY, _TINY),
    "7": lambda: time_forward(lambda: None, repeats=1),
}


def is_done(step: str) -> bool:
    """Call a step's function on a tiny input to see whether it still raises."""
    try:
        _PROBES[step]()
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        # Any other error means the student wrote something; let it surface
        # later with a real error message rather than being silently skipped.
        return True
    return True


def require(step: str, needs: str, purpose: str) -> None:
    """Raise NotImplementedError if `step` or any step in `needs` is unfinished.

    The current step is tried first so that its own TODO message is the one
    reported. A missing earlier step gets a "needs Step N" message instead.
    """
    _PROBES[step]()  # lets this step's own TODO message through unchanged
    for earlier in needs:
        if not is_done(earlier):
            raise NotImplementedError(f"needs Step {earlier} ({STEP_NAMES[earlier]}) to {purpose}")
