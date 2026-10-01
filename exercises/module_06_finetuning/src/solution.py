"""The `--solution` switch, provided for you.

You do NOT need to edit this file. `src/main.py` calls
`use_solution_if_requested()` before it imports anything from `exercise`.
With `--solution` on the command line, it loads solution/exercise.py and
registers it under the name "exercise", so every later
`from exercise import ...` picks up the finished answers instead of yours.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOLUTION_FILE = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"


def use_solution_if_requested() -> None:
    """Swap solution/exercise.py in for exercise.py when --solution is given."""
    if "--solution" not in sys.argv:
        return
    sys.argv.remove("--solution")  # so argparse in main.py never sees it
    spec = importlib.util.spec_from_file_location("exercise", _SOLUTION_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exercise"] = module
    spec.loader.exec_module(module)
