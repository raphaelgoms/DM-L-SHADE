#!/usr/bin/env python3
"""Compare L-SHADE and DM-L-SHADE from the result files of cec2014_collect.py.

For each function it reports the mean and standard deviation of the final error
(the last row of the 14 x 51 matrix, i.e. at MaxFES) of both algorithms and a
Wilcoxon rank-sum test between the two samples of runs, the test used to
compare algorithms in the CEC competitions. The last column says whether
DM-L-SHADE is significantly better (+), worse (-) or not different (=) from
L-SHADE. A summary with the counts follows each table.

    python3 experiments/compare_cec2014_results.py
    python3 experiments/compare_cec2014_results.py --dims 10 --alpha 0.01
"""

import argparse
import math
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE, CANDIDATE = "L-SHADE", "DM-L-SHADE"


def final_errors(results, name, function, dim):
    """Error at MaxFES for each run (last checkpoint row of the matrix)."""
    path = results / name / f"{name}_{function}_{dim}.txt"
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [float(value) for value in rows[-1]]


def rank_sum_test(x, y):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test, normal approximation
    with tie correction. Returns (p-value, mean rank of x, mean rank of y)."""
    n1, n2 = len(x), len(y)
    n = n1 + n2
    ordered = sorted([(v, 0) for v in x] + [(v, 1) for v in y])

    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2  # ranks i+1 .. j
        for k in range(i, j):
            ranks[k] = average_rank
        tie_term += (j - i) ** 3 - (j - i)
        i = j

    rank_sum_x = sum(rank for rank, (_, group) in zip(ranks, ordered) if group == 0)
    mean_rank_x = rank_sum_x / n1
    mean_rank_y = (n * (n + 1) / 2 - rank_sum_x) / n2

    u = rank_sum_x - n1 * (n1 + 1) / 2
    mean_u = n1 * n2 / 2
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance == 0:  # every value tied: the samples are indistinguishable
        return 1.0, mean_rank_x, mean_rank_y

    z = (abs(u - mean_u) - 0.5) / math.sqrt(variance)  # continuity correction
    p_value = math.erfc(max(z, 0.0) / math.sqrt(2))     # two-sided
    return p_value, mean_rank_x, mean_rank_y


def available_dims(results):
    pattern = re.compile(rf"^{re.escape(BASELINE)}_\d+_(\d+)\.txt$")
    dims = {int(m.group(1)) for f in (results / BASELINE).glob("*.txt") if (m := pattern.match(f.name))}
    return sorted(dims)


def available_functions(results, dim):
    functions = set(range(1, 31))
    for name in (BASELINE, CANDIDATE):
        functions &= {f for f in functions if (results / name / f"{name}_{f}_{dim}.txt").exists()}
    return sorted(functions)


def compare_dimension(results, dim, alpha):
    functions = available_functions(results, dim)
    if not functions:
        return

    print(f"\nD = {dim}: error at MaxFES, mean (std) over runs; "
          f"Wilcoxon rank-sum, alpha = {alpha}")
    print(f"{'Func.':>5}  {BASELINE + ' mean':>14} {'(std)':>13}  "
          f"{CANDIDATE + ' mean':>17} {'(std)':>13}  {'p-value':>9}  result")

    counts = {"+": 0, "-": 0, "=": 0}
    for function in functions:
        base = final_errors(results, BASELINE, function, dim)
        cand = final_errors(results, CANDIDATE, function, dim)
        p_value, rank_base, rank_cand = rank_sum_test(base, cand)

        if p_value < alpha:
            symbol = "+" if rank_cand < rank_base else "-"
        else:
            symbol = "="
        counts[symbol] += 1

        print(f"{function:>5}  {statistics.mean(base):>14.4e} ({statistics.pstdev(base):>10.2e})  "
              f"{statistics.mean(cand):>17.4e} ({statistics.pstdev(cand):>10.2e})  "
              f"{p_value:>9.4f}  {symbol}")

    print(f"\n{CANDIDATE} vs {BASELINE} on {len(functions)} functions: "
          f"{counts['+']} better (+), {counts['-']} worse (-), {counts['=']} no significant difference (=)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2014",
                        help="directory written by cec2014_collect.py")
    parser.add_argument("--dims", nargs="+", type=int,
                        help="dimensions to compare (default: all found)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level (default: 0.05)")
    args = parser.parse_args()

    dims = args.dims or available_dims(args.results)
    if not dims:
        raise SystemExit(f"no result files found in {args.results}; run cec2014_collect.py first")
    for dim in dims:
        compare_dimension(args.results, dim, args.alpha)


if __name__ == "__main__":
    main()
