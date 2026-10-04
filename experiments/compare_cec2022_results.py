#!/usr/bin/env python3
"""Compare L-SHADE and DM-L-SHADE from the result files of cec2022_collect.py.

Follows the evaluation criterion of the CEC-2022 rules (section 2.3), which
rewards speed as well as accuracy: every trial ends either at MaxFES or when
its error reaches 1e-8, so trials are ordered by their final error and, among
those that reached 1e-8, by the evaluations they needed (FE_term).

For each function it reports the mean final error and mean FE_term of both
algorithms, a Wilcoxon rank-sum test between the two samples of trials under
that ordering, and each algorithm's score: the sum of the ranks of its trials
(the best of all trials getting the highest rank) minus n(n+1)/2, which for
two algorithms is the Mann-Whitney U statistic. The result column says whether
DM-L-SHADE is significantly better (+), worse (-) or not different (=) from
L-SHADE. A summary with the counts and the final scores (sum over functions)
follows each table.

    python3 experiments/compare_cec2022_results.py
    python3 experiments/compare_cec2022_results.py --dims 10 --alpha 0.01
"""

import argparse
import math
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE, CANDIDATE = "L-SHADE", "DM-L-SHADE"
CHECKPOINTS = 16  # error rows in the result files; FE_term is the row after them


def trials(results, name, function, dim):
    """(final error, FE_term) of each run; sorting these orders trials best first."""
    path = results / name / f"{name}_{function}_{dim}.txt"
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    errors = [float(value) for value in rows[CHECKPOINTS - 1]]
    fe_terms = [int(float(value)) for value in rows[CHECKPOINTS]]
    return list(zip(errors, fe_terms))


def rank_sum_test(x, y):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test, normal approximation
    with tie correction. Returns (p-value, mean rank of x, mean rank of y),
    rank 1 being the smallest value."""
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


def score(x, y):
    """CEC-2022 score of x against y: the number of trials of y that each
    trial of x beats, ties counting half (sum of ranks minus n(n+1)/2)."""
    return sum((a < b) + 0.5 * (a == b) for a in x for b in y)


def available_dims(results):
    pattern = re.compile(rf"^{re.escape(BASELINE)}_\d+_(\d+)\.txt$")
    dims = {int(m.group(1)) for f in (results / BASELINE).glob("*.txt") if (m := pattern.match(f.name))}
    return sorted(dims)


def available_functions(results, dim):
    functions = set(range(1, 13))
    for name in (BASELINE, CANDIDATE):
        functions &= {f for f in functions if (results / name / f"{name}_{f}_{dim}.txt").exists()}
    return sorted(functions)


def compare_dimension(results, dim, alpha):
    functions = available_functions(results, dim)
    if not functions:
        return

    print(f"\nD = {dim}: mean error at the end of the run and mean FE_term; "
          f"Wilcoxon rank-sum, alpha = {alpha}")
    print(f"{'Func.':>5}  {BASELINE + ' error':>15} {'FEs':>9}  "
          f"{CANDIDATE + ' error':>18} {'FEs':>9}  {'p-value':>9}  result  "
          f"{'score ' + BASELINE:>14} {'score ' + CANDIDATE:>17}")

    counts = {"+": 0, "-": 0, "=": 0}
    totals = {BASELINE: 0.0, CANDIDATE: 0.0}
    for function in functions:
        base = trials(results, BASELINE, function, dim)
        cand = trials(results, CANDIDATE, function, dim)
        p_value, rank_base, rank_cand = rank_sum_test(base, cand)

        if p_value < alpha:
            symbol = "+" if rank_cand < rank_base else "-"
        else:
            symbol = "="
        counts[symbol] += 1

        score_base, score_cand = score(base, cand), score(cand, base)
        totals[BASELINE] += score_base
        totals[CANDIDATE] += score_cand

        print(f"{function:>5}  {statistics.mean(e for e, _ in base):>15.4e} "
              f"{statistics.mean(f for _, f in base):>9.0f}  "
              f"{statistics.mean(e for e, _ in cand):>18.4e} "
              f"{statistics.mean(f for _, f in cand):>9.0f}  "
              f"{p_value:>9.4f}  {symbol:^6}  {score_base:>14g} {score_cand:>17g}")

    print(f"\n{CANDIDATE} vs {BASELINE} on {len(functions)} functions: "
          f"{counts['+']} better (+), {counts['-']} worse (-), {counts['=']} no significant difference (=)")
    print(f"Final scores: {BASELINE} {totals[BASELINE]:g}, {CANDIDATE} {totals[CANDIDATE]:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2022",
                        help="directory written by cec2022_collect.py")
    parser.add_argument("--dims", nargs="+", type=int,
                        help="dimensions to compare (default: all found)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level (default: 0.05)")
    args = parser.parse_args()

    dims = args.dims or available_dims(args.results)
    if not dims:
        raise SystemExit(f"no result files found in {args.results}; run cec2022_collect.py first")
    for dim in dims:
        compare_dimension(args.results, dim, args.alpha)


if __name__ == "__main__":
    main()
