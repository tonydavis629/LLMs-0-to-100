"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Several steps call metrics from earlier
steps. If Step 3 were unfinished, running Step 5 would report Step 3's TODO
under Step 5's header. These helpers avoid that: each step first tries its
own blank on a tiny input, then checks the earlier steps it depends on, and
names the first one that is still missing.
"""

from __future__ import annotations

from exercise import normalize_answer


def require(step: str, call, why: str) -> None:
    """Stop this step with a pointed message if an earlier step is unfinished.

    `step` names the earlier step (for example "Step 3 (exact_match)") and
    `call` is a tiny call into it that raises NotImplementedError until it is done.
    """
    try:
        call()
    except NotImplementedError:
        raise NotImplementedError(f"needs {step} to {why}") from None


def probe_own(fn, *args, why: str) -> None:
    """Call exact_match() or token_f1() once on a tiny input, before the real work.

    Both call normalize_answer() (Step 2) before they reach their own blank. So
    the probe first swaps in a stand-in normalize_answer() that returns the text
    unchanged: if the function still stops with a TODO, the TODO is its own and
    that is what the step reports. Only then does it check Step 2 is finished.
    """
    names = fn.__globals__                          # the exercise module's global names
    real = names["normalize_answer"]
    names["normalize_answer"] = lambda text: text   # stand-in: text unchanged
    try:
        fn(*args)
    except NotImplementedError:
        raise                                       # the step's own blank is unfinished
    except Exception:  # noqa: BLE001
        pass                                        # other problems surface in the real run
    finally:
        names["normalize_answer"] = real            # always put the real one back
    require("Step 2 (normalize_answer)", lambda: normalize_answer("x"), why)
