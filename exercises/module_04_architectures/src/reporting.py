"""Printing each step's result, provided for you.

You do NOT need to edit this file. `run_step()` runs one step of the
exercise, then prints its header with a tag, the output your code produced,
and one line per test. The tags are:

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


class MissingPrerequisite(NotImplementedError):
    """An earlier step is unfinished, so this step's full-size output cannot run.

    The step's own tests use small fake models, so they still run.
    """


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

    `show()` prints whatever the student's code produces (shapes, generated
    text, saved plots). `check()` returns the list of Check results for the step.

    If `show()` needs an unfinished earlier step, its output is skipped but
    the step's own tests still run, so each blank gets feedback on its own.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        try:
            with redirect_stdout(buffer):
                show()
        except MissingPrerequisite as e:
            note = f"output skipped: {e}"
        try:
            checks = check()
        except NotImplementedError:
            if note:  # the tests also need the earlier step
                raise MissingPrerequisite(note.removeprefix("output skipped: ")) from None
            raise
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
