"""Printing each step's result, provided for you.

You do NOT need to edit this file. `run_step()` runs one step of the
exercise, then prints its header with a tag, the output your code produced,
and one line per test. `catch_up()` runs an earlier step's training quietly
when a later step is run on its own. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (shapes, training
    progress, generated captions). `check()` returns the list of Check results
    for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


def catch_up(train, summary) -> object:
    """Return `train()`'s result, running it quietly if no earlier step has.

    `train` is a training function wrapped in @cache, so it only ever runs once.
    When an earlier step already ran it (and showed its progress), this just
    returns the saved result. Otherwise it trains with the progress hidden and
    prints the one-line `summary(result)` instead, e.g. after `--step 7`.
    """
    if train.cache_info().currsize:  # already trained by an earlier step
        return train()
    with redirect_stdout(io.StringIO()):
        result = train()
    print(summary(result))
    return result
