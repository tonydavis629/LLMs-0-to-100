"""Step 7: time_forward()

Run by src/main.py. You do NOT need to edit this file.

Timings change from run to run, so these checks only test what must always
hold: the units, the sign, and the number of calls. None of them depends on
how fast your machine is.
"""

from __future__ import annotations

import time

from tests.check import Check, bad, ok


def check_time_forward(time_forward) -> list[Check]:
    """Elapsed time per run, in milliseconds, fastest of `repeats` runs."""
    checks = []

    # time.sleep(0.02) always takes at least 20 ms, so the answer must be at
    # least 20 in milliseconds. Seconds would come out as 0.02.
    ms = time_forward(lambda: time.sleep(0.02), repeats=2)
    checks.append(
        ok("reports milliseconds: a 20 ms sleep measures at least 20")
        if 19.5 <= ms < 5000
        else bad("reports milliseconds: a 20 ms sleep measures at least 20",
                 f"got {ms:.4f} (did you subtract start and multiply by 1000?)")
    )

    # A function that does nothing still takes a little time, never negative
    ms = time_forward(lambda: None)
    checks.append(
        ok("an empty function measures a small positive time (0 to 5 ms)")
        if 0 <= ms < 5
        else bad("an empty function measures a small positive time (0 to 5 ms)",
                 f"got {ms:.4f} (elapsed time is time.perf_counter() - start, in that order)")
    )

    # One untimed warm-up call, then one timed call per repeat
    calls = []
    time_forward(lambda: calls.append(1), repeats=3)
    checks.append(
        ok("calls fn once to warm up and once per repeat: repeats=3 makes 4 calls")
        if len(calls) == 4
        else bad("calls fn once to warm up and once per repeat: repeats=3 makes 4 calls",
                 f"fn was called {len(calls)} times")
    )
    return checks
