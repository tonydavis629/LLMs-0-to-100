"""
Module 12 Exercise runner: linear attention, two ways

Run with:
    uv run python module_12_future/src/main.py

Every step is tagged on its header line, then its output follows: whatever
your code produced and the result of each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-7).
Add --solution to run the finished answers from solution/exercise.py.

Steps 1-3 build the parallel form and compare it with the softmax attention
from Module 3. Steps 4-5 build the recurrent form and compare it with the
parallel form. Step 6 decides whether the two agree, and Step 7 times all
three implementations across sequence lengths, fits the exponent of each cost
curve, and saves a log-log plot.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided attention and plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

from exercise import (  # noqa: E402  (import after sys.path edits)
    feature_map,
    masked_scores,
    parallel_linear_attention,
    update_state,
    recurrent_step_output,
    outputs_match,
    time_forward,
)
from attention import softmax_attention, run_recurrent, make_inputs  # noqa: E402
from visualization import save_scaling_plot  # noqa: E402

# One test file per step lives in tests/
from tests.test_step1_feature_map import check_feature_map  # noqa: E402
from tests.test_step2_masked_scores import check_masked_scores  # noqa: E402
from tests.test_step3_parallel import (  # noqa: E402
    check_parallel_linear_attention,
    known_good_steps_1_and_2,
)
from tests.test_step4_update_state import check_update_state  # noqa: E402
from tests.test_step5_recurrent_output import check_recurrent_step_output  # noqa: E402
from tests.test_step6_outputs_match import check_outputs_match  # noqa: E402
from tests.test_step7_time_forward import check_time_forward  # noqa: E402


HEAD_DIM = 64                                     # query/key/value width
CHECK_LENGTH = 256                                # sequence length for the equivalence check
SWEEP_LENGTHS = [512, 1024, 2048, 4096, 8192]     # sequence lengths for the timing sweep
RECURRENT_LIMIT = 8192                            # skip the Python loop above this length

SOFTMAX_LABEL = "softmax attention (parallel)"
LINEAR_PARALLEL_LABEL = "linear attention (parallel)"
LINEAR_RECURRENT_LABEL = "linear attention (recurrent)"

_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
_OUTPUT_DIR = _MODULE_DIR / "output"


# ---------------------------------------------------------------------------
# Which steps are finished?
# ---------------------------------------------------------------------------

# Each step's function, and a way to call it on throwaway tensors
_TINY = torch.ones(2, 2)
_VEC = torch.ones(2)
_PROBES = {
    1: ("feature_map", lambda: feature_map(_TINY)),
    2: ("masked_scores", lambda: masked_scores(_TINY, _TINY)),
    3: ("parallel_linear_attention", lambda: parallel_linear_attention(_TINY, _TINY, _TINY)),
    4: ("update_state", lambda: update_state(_TINY, _VEC, _VEC, _VEC)),
    5: ("recurrent_step_output", lambda: recurrent_step_output(_VEC, _TINY, _VEC)),
    6: ("outputs_match", lambda: outputs_match(_TINY, _TINY)),
    7: ("time_forward", lambda: time_forward(lambda: None, repeats=1)),
}


def _is_done(step: int) -> bool:
    """Call a step's function on a tiny input to see whether it still raises."""
    try:
        _PROBES[step][1]()
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        # Any other error means the student wrote something; let it surface
        # later with a real error message rather than being silently skipped.
        return True
    return True


def _require(steps: list[int], purpose: str) -> None:
    """Stop a step's demo with a pointed message if an earlier step is unfinished.

    Without this, the demo would stop on the earlier step's own TODO message,
    which reads as if THIS step were the problem.
    """
    for step in steps:
        if not _is_done(step):
            name = _PROBES[step][0]
            raise NotImplementedError(f"needs Step {step} ({name}) to {purpose}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _fit_slope(lengths: list[int], timings: list[float]) -> float:
    """Fit the exponent of a power law by least squares in log-log space.

    If time = c * n^p then log(time) = log(c) + p * log(n), so the slope of the
    fitted line IS the complexity exponent. This is the same log-log fit the
    scaling-law literature uses on loss curves, applied to runtime.
    """
    xs = [math.log(n) for n in lengths]
    ys = [math.log(t) for t in timings]
    n_points = len(xs)
    mean_x = sum(xs) / n_points
    mean_y = sum(ys) / n_points
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance = sum((x - mean_x) ** 2 for x in xs)
    return covariance / variance


def _build_callables(done: dict[int, bool], n: int) -> dict[str, object]:
    """Zero-argument callables for whichever implementations are ready."""
    Q, K, V = make_inputs(n, HEAD_DIM)
    runnable: dict[str, object] = {SOFTMAX_LABEL: lambda: softmax_attention(Q, K, V)}
    if done[1] and done[2] and done[3]:
        runnable[LINEAR_PARALLEL_LABEL] = lambda: parallel_linear_attention(Q, K, V)
    if done[1] and done[4] and done[5] and n <= RECURRENT_LIMIT:
        runnable[LINEAR_RECURRENT_LABEL] = lambda: run_recurrent(
            Q, K, V, feature_map, update_state, recurrent_step_output
        )
    return runnable


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

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

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

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


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1() -> str:
    """The positive feature map, on a few inputs from negative to positive."""

    def show():
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        values = ", ".join(f"{v:.4f}" for v in feature_map(x).flatten().tolist())
        print(f"  feature_map([-3, -1, 0, 1, 3]) = [{values}]")

    return run_step("Step 1: feature_map()", show, lambda: check_feature_map(feature_map))


def step_2() -> str:
    """The masked score matrix for 4 tokens, small enough to read."""

    def show():
        feats = torch.tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
        scores = masked_scores(feats, feats)
        print("  4 tokens, q_phi = k_phi = [[1,1],[1,2],[2,1],[2,2]]")
        print("  row i is query i, column j is key j:")
        for row in scores.tolist():
            print("    " + "".join(f"{v:6.1f}" for v in row))

    return run_step("Step 2: masked_scores()", show, lambda: check_masked_scores(masked_scores))


def step_3() -> str:
    """The parallel form on the demo input, compared with softmax attention."""

    def show():
        # Your own Step 3 line first, with known-good Steps 1 and 2 swapped
        # in, so an unfinished Step 3 reports its own TODO
        with known_good_steps_1_and_2(parallel_linear_attention):
            parallel_linear_attention(_TINY, _TINY, _TINY)
        _require([1, 2], "compute the parallel form")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        parallel = parallel_linear_attention(Q, K, V)
        print(f"  parallel form: {CHECK_LENGTH} tokens, head dimension {HEAD_DIM}, "
              f"output shape {tuple(parallel.shape)}")
        # Linear vs softmax: genuinely different functions, so expect a real gap
        gap = (parallel - softmax_attention(Q, K, V)).abs().max().item()
        print(f"  linear vs softmax        max difference = {gap:.2e}")

    return run_step("Step 3: parallel_linear_attention()", show,
                    lambda: check_parallel_linear_attention(parallel_linear_attention))


def step_4() -> str:
    return run_step("Step 4: update_state()", lambda: None,
                    lambda: check_update_state(update_state))


def step_5() -> str:
    """The recurrent form on the same input, compared with the parallel form."""

    def show():
        # Your own Step 5 line first, so an unfinished Step 5 reports its own TODO
        recurrent_step_output(_VEC, _TINY, _VEC)
        _require([1, 2, 3, 4], "compare the recurrent form with the parallel form")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        recurrent = run_recurrent(Q, K, V, feature_map, update_state, recurrent_step_output)
        print(f"  recurrent form: {CHECK_LENGTH} tokens one at a time, "
              f"output shape {tuple(recurrent.shape)}")
        # The two linear forms: algebraically identical, so any gap is float noise
        gap = (parallel_linear_attention(Q, K, V) - recurrent).abs().max().item()
        print(f"  parallel vs recurrent    max difference = {gap:.2e}")

    return run_step("Step 5: recurrent_step_output()", show,
                    lambda: check_recurrent_step_output(recurrent_step_output))


def step_6() -> str:
    """Your outputs_match() decides both comparisons from Steps 3 and 5."""

    def show():
        # Your own Step 6 line first, so an unfinished Step 6 reports its own TODO
        outputs_match(_TINY, _TINY)
        _require([1, 2, 3, 4, 5], "compare the two forms")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        parallel = parallel_linear_attention(Q, K, V)
        recurrent = run_recurrent(Q, K, V, feature_map, update_state, recurrent_step_output)
        softmax_out = softmax_attention(Q, K, V)

        linear_gap = (parallel - recurrent).abs().max().item()
        softmax_gap = (parallel - softmax_out).abs().max().item()
        linear_verdict = "MATCH" if outputs_match(parallel, recurrent) else "MISMATCH"
        softmax_verdict = "MATCH" if outputs_match(parallel, softmax_out) else "DIFFERENT"
        print(f"  parallel vs recurrent    max difference = {linear_gap:.2e}   {linear_verdict}")
        print(f"  linear   vs softmax      max difference = {softmax_gap:.2e}   {softmax_verdict}")
        print()
        print("  The first line is the theorem: the quadratic form and the RNN")
        print("  compute the same function, and differ only in floating-point noise.")
        print("  The second line is the caveat: linear attention is a DIFFERENT")
        print("  function from softmax attention, not an approximation of it.")

    return run_step("Step 6: outputs_match()", show, lambda: check_outputs_match(outputs_match))


def step_7() -> str:
    """Time every ready implementation at every length, fit the exponents, plot."""

    def show():
        # Softmax attention is provided, so it always runs. The two linear
        # forms join the table once their steps are finished.
        done = {step: _is_done(step) for step in _PROBES}
        labels = [SOFTMAX_LABEL, LINEAR_PARALLEL_LABEL, LINEAR_RECURRENT_LABEL]
        results: dict[str, list] = {label: [] for label in labels}

        print(f"  {'':>6}{'softmax':>13}{'linear':>13}{'linear':>13}")
        print(f"  {'n':>6}{'parallel':>13}{'parallel':>13}{'recurrent':>13}")
        # The leading "+" keeps this rule from being read as a Markdown
        # heading when the output is pasted into the slides
        print("  +" + "-" * 44)
        for n in SWEEP_LENGTHS:
            runnable = _build_callables(done, n)
            row = f"  {n:>6}"
            for label in labels:
                fn = runnable.get(label)
                if fn is None:
                    results[label].append(None)
                    row += f"{'-':>13}"
                else:
                    ms = time_forward(fn)
                    results[label].append(ms)
                    row += f"{ms:>10.2f} ms"
            print(row)
        print()
        print("  Times are milliseconds per forward pass, best of 3.")
        if any(t is not None for t in results[LINEAR_RECURRENT_LABEL]):
            print("  Read the columns from top to bottom, not left to right. The recurrent")
            print("  form starts an order of magnitude slower than either parallel form and")
            print("  ends up the fastest of the three, because it is the only one whose cost")
            print("  is not growing with the square of the sequence length.")

        # The exponent of each cost curve: 2 means quadratic, 1 means linear
        print()
        print("  Fitted slope of log(time) against log(n). This IS the exponent in")
        print("  the big-O: 2 means quadratic, 1 means linear.")
        print()
        all_positive = True
        for label, timings in results.items():
            pairs = [(n, t) for n, t in zip(SWEEP_LENGTHS, timings) if t is not None]
            if len(pairs) < 2:
                continue
            if any(t <= 0 for _, t in pairs):
                # The log of a zero or negative time is undefined
                all_positive = False
                print(f"  {label:<32} slope = ? (a time was zero or negative)")
                continue
            slope = _fit_slope([n for n, _ in pairs], [t for _, t in pairs])
            print(f"  {label:<32} slope = {slope:.2f}")
        if any(t is not None for t in results[LINEAR_PARALLEL_LABEL]):
            print()
            print("  Both parallel forms scale quadratically: the n-by-n score matrix is")
            print("  the cost, and removing the softmax does not remove the matrix.")
            if any(t is not None for t in results[LINEAR_RECURRENT_LABEL]):
                print("  Only the recurrent form escapes it, because it never builds one.")

        # The log-log plot of the same numbers (log axes need positive times)
        print()
        if all_positive:
            save_scaling_plot(SWEEP_LENGTHS, results, _OUTPUT_DIR / "attention_scaling.png")
            print("  Saved figure to output/attention_scaling.png")
        else:
            print("  Figure skipped: a log-log plot needs every time to be positive.")

    return run_step("Step 7: time_forward()", show, lambda: check_time_forward(time_forward))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4,
         "5": step_5, "6": step_6, "7": step_7}


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear attention, two ways")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = list(STEPS) if args.step == "all" else [args.step]

    for step in steps:
        STEPS[step]()


if __name__ == "__main__":
    main()
