"""
Module 12 Exercise runner: linear attention, two ways

Run with:
    uv run python module_12_future/src/main.py

Runs three attention implementations over the same random inputs: the softmax
attention from Module 3, linear attention computed the quadratic way, and
linear attention computed as an RNN. It first checks whether the two linear
forms agree (they should, exactly; that is the theorem), then times all three
across sequence lengths and fits the exponent of each cost curve.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one function at a time and re-run immediately.
The parallel form works after step 3, the recurrent form joins after step 5,
the equivalence check after step 6, and the timing sweep after step 7.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided attention and plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


HEAD_DIM = 64                                     # query/key/value width
CHECK_LENGTH = 256                                # sequence length for the equivalence check
SWEEP_LENGTHS = [512, 1024, 2048, 4096, 8192]     # sequence lengths for the timing sweep
RECURRENT_LIMIT = 8192                            # skip the Python loop above this length

SOFTMAX_LABEL = "softmax attention (parallel)"
LINEAR_PARALLEL_LABEL = "linear attention (parallel)"
LINEAR_RECURRENT_LABEL = "linear attention (recurrent)"

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR.parent / "output"


def _is_implemented(fn, *args, **kwargs) -> bool:
    """Call a student function on a tiny input to see whether it still raises."""
    try:
        fn(*args, **kwargs)
    except NotImplementedError:
        return False
    except Exception:
        # Any other error means the student wrote something; let it surface
        # later with a real traceback rather than being silently skipped.
        return True
    return True


def _probe_steps() -> dict[str, bool]:
    """Work out which steps are done, using throwaway tensors."""
    tiny = torch.ones(2, 2)
    vec = torch.ones(2)
    return {
        "feature_map": _is_implemented(feature_map, tiny),
        "masked_scores": _is_implemented(masked_scores, tiny, tiny),
        "parallel_linear_attention": _is_implemented(
            parallel_linear_attention, tiny, tiny, tiny
        ),
        "update_state": _is_implemented(update_state, tiny, vec, vec, vec),
        "recurrent_step_output": _is_implemented(recurrent_step_output, vec, tiny, vec),
        "outputs_match": _is_implemented(outputs_match, tiny, tiny),
        "time_forward": _is_implemented(time_forward, lambda: None, repeats=1),
    }


def _heading(title: str) -> None:
    """Print a section heading.

    The leading "+" on rule lines is deliberate. A line of nothing but dashes
    directly under a line of text is heading syntax in Markdown, and this
    output gets pasted into the lecture slides verbatim.
    """
    print()
    print(f"+-- {title} " + "-" * max(0, 62 - len(title)))


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


def check_equivalence(done: dict[str, bool]) -> None:
    """Do the two forms of linear attention compute the same function?"""
    _heading("Do the two forms agree?")
    print(f"  sequence length {CHECK_LENGTH}, head dimension {HEAD_DIM}")
    print()

    Q, K, V = make_inputs(CHECK_LENGTH, HEAD_DIM)

    if not done["parallel_linear_attention"]:
        print("  Skipped: the parallel form needs steps 1-3.")
        return
    parallel = parallel_linear_attention(Q, K, V)

    if not done["update_state"] or not done["recurrent_step_output"]:
        print("  The parallel form runs. The recurrent form needs steps 4-5.")
        return
    recurrent = run_recurrent(
        Q, K, V, feature_map, update_state, recurrent_step_output
    )

    # The two linear forms: algebraically identical, so any gap is float noise.
    linear_gap = (parallel - recurrent).abs().max().item()
    # Linear vs softmax: genuinely different functions, so expect a real gap.
    softmax_out = softmax_attention(Q, K, V)
    softmax_gap = (parallel - softmax_out).abs().max().item()

    if done["outputs_match"]:
        verdict = "MATCH" if outputs_match(parallel, recurrent) else "MISMATCH"
    else:
        verdict = "(step 6 not implemented)"

    print(f"  parallel vs recurrent    max difference = {linear_gap:.2e}   {verdict}")
    print(f"  linear   vs softmax      max difference = {softmax_gap:.2e}   DIFFERENT")
    print()
    print("  The first line is the theorem: the quadratic form and the RNN")
    print("  compute the same function, and differ only in floating-point noise.")
    print("  The second line is the caveat: linear attention is a DIFFERENT")
    print("  function from softmax attention, not an approximation of it.")


def _build_callables(done: dict[str, bool], n: int) -> dict[str, object]:
    """Zero-argument callables for whichever implementations are ready."""
    Q, K, V = make_inputs(n, HEAD_DIM)
    runnable: dict[str, object] = {SOFTMAX_LABEL: lambda: softmax_attention(Q, K, V)}
    if done["parallel_linear_attention"]:
        runnable[LINEAR_PARALLEL_LABEL] = lambda: parallel_linear_attention(Q, K, V)
    if done["update_state"] and done["recurrent_step_output"] and n <= RECURRENT_LIMIT:
        runnable[LINEAR_RECURRENT_LABEL] = lambda: run_recurrent(
            Q, K, V, feature_map, update_state, recurrent_step_output
        )
    return runnable


def run_timing_sweep(done: dict[str, bool]) -> dict[str, list]:
    """Time every ready implementation at every sequence length."""
    _heading("What does it cost?")

    if not done["time_forward"]:
        print("  Skipped: the timing sweep needs step 7.")
        return {}

    labels = [SOFTMAX_LABEL, LINEAR_PARALLEL_LABEL, LINEAR_RECURRENT_LABEL]
    results: dict[str, list] = {label: [] for label in labels}

    print(f"  {'':>6}{'softmax':>13}{'linear':>13}{'linear':>13}")
    print(f"  {'n':>6}{'parallel':>13}{'parallel':>13}{'recurrent':>13}")
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
    return results


def report_slopes(results: dict[str, list]) -> None:
    """Fit and print the complexity exponent of each cost curve."""
    if not results:
        return
    _heading("What are the exponents?")
    print("  Fitted slope of log(time) against log(n). This IS the exponent in")
    print("  the big-O: 2 means quadratic, 1 means linear.")
    print()
    for label, timings in results.items():
        pairs = [(n, t) for n, t in zip(SWEEP_LENGTHS, timings) if t is not None]
        if len(pairs) < 2:
            continue
        slope = _fit_slope([n for n, _ in pairs], [t for _, t in pairs])
        print(f"  {label:<32} slope = {slope:.2f}")
    print()
    print("  Both parallel forms scale quadratically: the n-by-n score matrix is")
    print("  the cost, and removing the softmax does not remove the matrix.")
    print("  Only the recurrent form escapes it, because it never builds one.")


def main() -> None:
    print("+" + "=" * 65)
    print("|  Module 12: Linear attention, two ways")
    print("+" + "=" * 65)

    done = _probe_steps()
    finished = sum(done.values())
    print(f"  Steps implemented: {finished} of {len(done)}")
    if finished < len(done):
        pending = [name for name, ok in done.items() if not ok]
        print(f"  Still to do: {', '.join(pending)}")

    check_equivalence(done)
    results = run_timing_sweep(done)
    report_slopes(results)

    if results:
        out_path = _OUTPUT_DIR / "attention_scaling.png"
        save_scaling_plot(SWEEP_LENGTHS, results, out_path)
        print()
        print(f"Saved figure: {out_path}")

    print()


if __name__ == "__main__":
    main()
