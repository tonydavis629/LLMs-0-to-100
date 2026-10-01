"""Checking which steps are finished, provided for you.

You do NOT need to edit this file. Some demos call the code from several
steps at once: the Step 8 benchmark builds requests (Step 4), reads streams
(Step 6) and times them (Step 7). Two helpers keep the output pointed:

- `require(step)` tries the step's own function on a throwaway input first,
  so an unfinished step reports its own TODO before anything goes to the
  server, instead of waiting on the network.
- `missing_steps(...)` names the first earlier step a demo needs that is
  still unfinished, so the demo can print one "skipped" line instead.
"""

from __future__ import annotations

from exercise import (
    build_chat_request,
    decode_tokens_per_second,
    extract_reply,
    latency_stats,
    model_weight_bytes,
    parse_stream_line,
    total_throughput,
)

# One throwaway call per step. Each raises NotImplementedError while that
# step's blank is unfinished (and only then).
_PROBES = {
    "1": lambda: model_weight_bytes(1000, 16),
    "3": lambda: decode_tokens_per_second(1e12, 1e9),
    "4": lambda: build_chat_request("m", "hi", 0.7, 8),
    "5": lambda: extract_reply({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}),
    "6": lambda: parse_stream_line('data: {"choices": [{"delta": {"content": "hi"}}]}'),
    "7": lambda: latency_stats(0.0, [0.5, 0.6, 0.7]),
    "8": lambda: total_throughput([10, 10], 2.0),
}

# What each earlier step is needed for, used in "needs Step N (...)" messages
_NEEDED_FOR = {
    "1": "Step 1 (model_weight_bytes) to size the weights",
    "4": "Step 4 (build_chat_request) to build the request",
    "6": "Step 6 (parse_stream_line) to read the token stream",
    "7": "Step 7 (latency_stats) to time each stream",
}


def _unfinished(step: str) -> bool:
    """True if the step's function still raises NotImplementedError."""
    try:
        _PROBES[step]()
    except NotImplementedError:
        return True
    except Exception:  # noqa: BLE001 - it runs, so a wrong answer is for the tests to explain
        pass
    return False


def require(step: str) -> None:
    """Raise this step's own NotImplementedError if its blank is unfinished."""
    if _unfinished(step):
        _PROBES[step]()  # call again to let the TODO message through


def missing_steps(*needed: str) -> str | None:
    """Return "needs Step N (...)" for the first unfinished step in `needed`, else None."""
    for step in needed:
        if _unfinished(step):
            return f"needs {_NEEDED_FOR[step]}"
    return None
