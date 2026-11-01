#!/usr/bin/env python3
"""Compare the L-SHADE / DM-L-SHADE results of cec2026_collect.py.

Implements the score of section 1 of the CEC-2026 criteria (U-score; Price,
Kumar and Suganthan, 2023). Every trial of every algorithm meets every other
trial on a function, and each meeting gives out two points:

  * accuracy: to the trial with the lower final error (half each if equal);
  * speed: to the trial that first reached the higher of the two final errors,
    looking at the recording points (half each if at the same point).

An algorithm's score is the sum over its trials and all functions. Trials of the
same algorithm meet too, which adds a constant, n(n-1) per function. Also prints
a two-sided Wilcoxon rank-sum test on the final errors, as in the other suites.

Errors are rounded to 6 significant digits before comparing, the precision of the
coarsest entries' files. Equal errors decide many meetings (several functions have
plateaus at the same local optimum), so without a common precision floating-point
noise such as 300.0000000000009 against 300 would decide them.

    python3 experiments/compare_cec2026_results.py
    python3 experiments/compare_cec2026_results.py --dims 10 --alpha 0.01
"""

import argparse
import statistics
from bisect import bisect_left
from pathlib import Path

from compare_cec2014_results import rank_sum_test

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE, CANDIDATE = "L-SHADE", "DM-L-SHADE"
FUNCTIONS = list(range(1, 30))
ZERO_THRESHOLD = 1e-8
SIGNIFICANT_DIGITS = 6


def prepare_trials(matrix, digits=SIGNIFICANT_DIGITS):
    """Trials of a recording points x runs matrix, as (final error, history). The
    history is the negated best-so-far error, ascending, so that bisect_left
    finds the first recording point that reached an error. Values below 1e-8
    count as zero; the others are rounded to `digits` significant digits (None
    keeps them as they are)."""
    trials = []
    for run in range(len(matrix[0])):
        best = float("inf")
        history = []
        for row in matrix:
            error = 0.0 if row[run] < ZERO_THRESHOLD else row[run]
            if digits and error < best:
                error = float(f"{error:.{digits - 1}e}")
            best = min(best, error)
            history.append(-best)
        trials.append((best, history))
    return trials


def meeting_points(trials_a, trials_b):
    """Accuracy and speed points of the trials of a against those of b. The
    trials of b get len(a) * len(b) minus each of them."""
    accuracy = speed = 0.0
    for final_a, history_a in trials_a:
        for final_b, history_b in trials_b:
            accuracy += 1.0 if final_a < final_b else 0.5 if final_a == final_b else 0.0
            target = -max(final_a, final_b)
            reached_a = bisect_left(history_a, target)
            reached_b = bisect_left(history_b, target)
            speed += 1.0 if reached_a < reached_b else 0.5 if reached_a == reached_b else 0.0
    return accuracy, speed


def within_points(trials):
    """Points the trials of one algorithm give each other, in each of accuracy and speed."""
    return len(trials) * (len(trials) - 1) / 2


def read_matrix(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def compare_dimension(results, dim, alpha):
    path = lambda name, f: results / name / f"{name}_{f}_{dim}.txt"
    functions = [f for f in FUNCTIONS if path(BASELINE, f).exists() and path(CANDIDATE, f).exists()]
    if not functions:
        return
    print(f"\nD = {dim}, U-score (accuracy + speed) and rank-sum test at alpha = {alpha}")
    print(f"{'F':>3} {'L-SHADE mean':>14} {'DM-L-SHADE mean':>16} {'U L-SHADE':>10} {'U DM-L-SHADE':>13} {'p-value':>9}  result")
    totals = {BASELINE: [0.0, 0.0], CANDIDATE: [0.0, 0.0]}
    counts = {"+": 0, "=": 0, "-": 0}
    for f in functions:
        matrix = {name: read_matrix(path(name, f)) for name in (BASELINE, CANDIDATE)}
        trials = {name: prepare_trials(matrix[name]) for name in matrix}
        accuracy, speed = meeting_points(trials[CANDIDATE], trials[BASELINE])
        meetings = len(trials[CANDIDATE]) * len(trials[BASELINE])
        within = within_points(trials[CANDIDATE])
        candidate = (accuracy + within, speed + within)
        baseline = (meetings - accuracy + within, meetings - speed + within)
        totals[CANDIDATE] = [t + v for t, v in zip(totals[CANDIDATE], candidate)]
        totals[BASELINE] = [t + v for t, v in zip(totals[BASELINE], baseline)]

        finals = {name: matrix[name][-1] for name in matrix}
        finals = {name: [0.0 if v < ZERO_THRESHOLD else v for v in values] for name, values in finals.items()}
        p_value, rank_base, rank_cand = rank_sum_test(finals[BASELINE], finals[CANDIDATE])
        symbol = ("+" if rank_cand < rank_base else "-") if p_value < alpha else "="
        counts[symbol] += 1
        print(f"{f:>3} {statistics.mean(finals[BASELINE]):>14.4e} {statistics.mean(finals[CANDIDATE]):>16.4e} "
              f"{sum(baseline):>10.1f} {sum(candidate):>13.1f} {p_value:>9.4f}  {symbol}")
    print(f"DM-L-SHADE against L-SHADE: {counts['+']} better, {counts['=']} no difference, {counts['-']} worse")
    for name in (BASELINE, CANDIDATE):
        print(f"{name:>11}: accuracy {totals[name][0]:.1f} + speed {totals[name][1]:.1f} = U-score {sum(totals[name]):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compare L-SHADE and DM-L-SHADE under the CEC-2026 score.")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2026",
                        help="directory written by cec2026_collect.py")
    parser.add_argument("--dims", nargs="+", type=int, default=[10, 30, 50, 100],
                        help="dimensions to compare (default: whichever exist)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level (default: 0.05)")
    args = parser.parse_args()
    for dim in args.dims:
        compare_dimension(args.results, dim, args.alpha)


if __name__ == "__main__":
    main()
