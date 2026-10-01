"""
Module 12 Exercise runner: linear attention, two ways

Run with:
    uv run python module_12_future/src/main.py

Each step below runs one function you write in exercise.py, then tests it.
Read top to bottom, the steps build linear attention twice and compare:

    1. the positive feature map that replaces the exponential
    2. the causally masked score matrix
    3. the parallel form (compared with Module 3's softmax attention)
    4. the recurrent form's running state update
    5. the recurrent form's per-token output (compared with the parallel form)
    6. deciding whether two outputs agree
    7. timing all three implementations as the sequence grows

Add --step N to run one step (1-7).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    feature_map,
    masked_scores,
    outputs_match,
    parallel_linear_attention,
    recurrent_step_output,
    time_forward,
    update_state,
)
from src.attention import make_inputs, run_recurrent, softmax_attention
from src.prerequisites import is_done, require
from src.reporting import run_step
from src.scaling import print_fitted_slopes, print_timing_table
from src.visualization import save_scaling_plot

# One test file per step lives in tests/
from tests.test_step1_feature_map import check_feature_map
from tests.test_step2_masked_scores import check_masked_scores
from tests.test_step3_parallel import check_parallel_linear_attention
from tests.test_step4_update_state import check_update_state
from tests.test_step5_recurrent_output import check_recurrent_step_output
from tests.test_step6_outputs_match import check_outputs_match
from tests.test_step7_time_forward import check_time_forward

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

HEAD_DIM = 64                                     # query/key/value width
CHECK_LENGTH = 256                                # sequence length for the equivalence check
SWEEP_LENGTHS = [512, 1024, 2048, 4096, 8192]     # sequence lengths for the timing sweep
RECURRENT_LIMIT = 8192                            # skip the Python loop above this length

SOFTMAX_LABEL = "softmax attention (parallel)"
LINEAR_PARALLEL_LABEL = "linear attention (parallel)"
LINEAR_RECURRENT_LABEL = "linear attention (recurrent)"


def run_recurrent_form(Q, K, V):
    """Feed the tokens through your Step 4 and 5 functions one at a time."""
    return run_recurrent(Q, K, V, feature_map, update_state, recurrent_step_output)


# ---------------------------------------------------------------------------
# Step 1: the positive feature map
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        values = ", ".join(f"{v:.4f}" for v in feature_map(x).flatten().tolist())
        print(f"  feature_map([-3, -1, 0, 1, 3]) = [{values}]")

    return run_step("Step 1: feature_map()", show, lambda: check_feature_map(feature_map))


# ---------------------------------------------------------------------------
# Step 2: the causally masked score matrix
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        feats = torch.tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])  # 4 tokens
        scores = masked_scores(feats, feats)
        print("  4 tokens, q_phi = k_phi = [[1,1],[1,2],[2,1],[2,2]]")
        print("  row i is query i, column j is key j:")
        for row in scores.tolist():
            print("    " + "".join(f"{v:6.1f}" for v in row))

    return run_step("Step 2: masked_scores()", show, lambda: check_masked_scores(masked_scores))


# ---------------------------------------------------------------------------
# Step 3: the parallel form, compared with softmax attention
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require("3", needs="12", purpose="compute the parallel form")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        parallel = parallel_linear_attention(Q, K, V)
        print(f"  parallel form: {CHECK_LENGTH} tokens, head dimension {HEAD_DIM}, "
              f"output shape {tuple(parallel.shape)}")
        # Linear vs softmax: genuinely different functions, so expect a real gap
        gap = (parallel - softmax_attention(Q, K, V)).abs().max().item()
        print(f"  linear vs softmax        max difference = {gap:.2e}")

    return run_step("Step 3: parallel_linear_attention()", show,
                    lambda: check_parallel_linear_attention(parallel_linear_attention))


# ---------------------------------------------------------------------------
# Step 4: the recurrent form's state update (tests only; Step 5 uses it)
# ---------------------------------------------------------------------------


def step_4() -> str:
    return run_step("Step 4: update_state()", lambda: None,
                    lambda: check_update_state(update_state))


# ---------------------------------------------------------------------------
# Step 5: the recurrent form, compared with the parallel form
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        require("5", needs="1234", purpose="compare the recurrent form with the parallel form")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        recurrent = run_recurrent_form(Q, K, V)
        print(f"  recurrent form: {CHECK_LENGTH} tokens one at a time, "
              f"output shape {tuple(recurrent.shape)}")
        # The two linear forms: algebraically identical, so any gap is float noise
        gap = (parallel_linear_attention(Q, K, V) - recurrent).abs().max().item()
        print(f"  parallel vs recurrent    max difference = {gap:.2e}")

    return run_step("Step 5: recurrent_step_output()", show,
                    lambda: check_recurrent_step_output(recurrent_step_output))


# ---------------------------------------------------------------------------
# Step 6: do the outputs agree?
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        require("6", needs="12345", purpose="compare the two forms")
        Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)
        parallel = parallel_linear_attention(Q, K, V)
        recurrent = run_recurrent_form(Q, K, V)
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


# ---------------------------------------------------------------------------
# Step 7: timing all three implementations as the sequence grows
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        # Softmax attention is provided, so it always runs. The two linear
        # forms join the table once their steps are finished.
        parallel_ready = all(is_done(step) for step in "123")
        recurrent_ready = all(is_done(step) for step in "145")

        # One time (or None, if not run) per implementation per length
        timings = {SOFTMAX_LABEL: [], LINEAR_PARALLEL_LABEL: [], LINEAR_RECURRENT_LABEL: []}
        for n in SWEEP_LENGTHS:
            Q, K, V = make_inputs(n, HEAD_DIM)
            runs = {
                SOFTMAX_LABEL: lambda: softmax_attention(Q, K, V),
                LINEAR_PARALLEL_LABEL: lambda: parallel_linear_attention(Q, K, V),
                LINEAR_RECURRENT_LABEL: lambda: run_recurrent_form(Q, K, V),
            }
            if not parallel_ready:
                runs[LINEAR_PARALLEL_LABEL] = None
            if not recurrent_ready or n > RECURRENT_LIMIT:
                runs[LINEAR_RECURRENT_LABEL] = None
            for label, run in runs.items():
                timings[label].append(None if run is None else time_forward(run))

        # The table, then the exponent of each cost curve: 2 means quadratic, 1 means linear
        print_timing_table(SWEEP_LENGTHS, timings)
        all_positive = print_fitted_slopes(SWEEP_LENGTHS, timings)

        # The log-log plot of the same numbers (log axes need positive times)
        print()
        if all_positive:
            save_scaling_plot(SWEEP_LENGTHS, timings, OUTPUT_DIR / "attention_scaling.png")
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

    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
