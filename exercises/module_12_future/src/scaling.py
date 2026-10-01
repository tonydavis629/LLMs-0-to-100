"""Printing the Step 7 timing results, provided for you.

You do NOT need to edit this file. `src/main.py` times each attention
implementation at each sequence length, then hands the numbers here:

  print_timing_table   one row per sequence length, one column per implementation
  print_fitted_slopes  the exponent of each cost curve, fitted in log-log space
"""

from __future__ import annotations

import math


def fit_slope(lengths: list[int], timings: list[float]) -> float:
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


def print_timing_table(lengths: list[int], timings: dict[str, list]) -> None:
    """Print milliseconds per forward pass, with "-" where a form was not run.

    `timings` maps each implementation's label to one time per length (or None),
    in the column order softmax parallel, linear parallel, linear recurrent.
    """
    columns = list(timings.values())
    print(f"  {'':>6}{'softmax':>13}{'linear':>13}{'linear':>13}")
    print(f"  {'n':>6}{'parallel':>13}{'parallel':>13}{'recurrent':>13}")
    # The leading "+" keeps this rule from being read as a Markdown
    # heading when the output is pasted into the slides
    print("  +" + "-" * 44)
    for i, n in enumerate(lengths):
        row = f"  {n:>6}"
        for column in columns:
            ms = column[i]
            row += f"{'-':>13}" if ms is None else f"{ms:>10.2f} ms"
        print(row)
    print()
    print("  Times are milliseconds per forward pass, best of 3.")
    recurrent = columns[2]
    if any(t is not None for t in recurrent):
        print("  Read the columns from top to bottom, not left to right. The recurrent")
        print("  form starts an order of magnitude slower than either parallel form and")
        print("  ends up the fastest of the three, because it is the only one whose cost")
        print("  is not growing with the square of the sequence length.")


def print_fitted_slopes(lengths: list[int], timings: dict[str, list]) -> bool:
    """Print the fitted exponent for each implementation that ran at 2+ lengths.

    Returns False if any time was zero or negative (its log is undefined),
    which also rules out the log-log plot.
    """
    print()
    print("  Fitted slope of log(time) against log(n). This IS the exponent in")
    print("  the big-O: 2 means quadratic, 1 means linear.")
    print()
    all_positive = True
    for label, column in timings.items():
        pairs = [(n, t) for n, t in zip(lengths, column) if t is not None]
        if len(pairs) < 2:
            continue
        if any(t <= 0 for _, t in pairs):
            all_positive = False
            print(f"  {label:<32} slope = ? (a time was zero or negative)")
            continue
        slope = fit_slope([n for n, _ in pairs], [t for _, t in pairs])
        print(f"  {label:<32} slope = {slope:.2f}")

    _, linear_parallel, linear_recurrent = timings.values()
    if any(t is not None for t in linear_parallel):
        print()
        print("  Both parallel forms scale quadratically: the n-by-n score matrix is")
        print("  the cost, and removing the softmax does not remove the matrix.")
        if any(t is not None for t in linear_recurrent):
            print("  Only the recurrent form escapes it, because it never builds one.")
    return all_positive
