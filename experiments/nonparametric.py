"""Nonparametric tests for comparing algorithms, standard library only.

  * rank_sum_test: two-sided Wilcoxon rank-sum (Mann-Whitney U) test between
    the runs of two algorithms on one problem (from compare_cec2014_results);
  * signed_rank_test: two-sided Wilcoxon signed-rank test between two
    algorithms paired by problem (e.g. mean gap per instance);
  * friedman_test: Friedman test of k algorithms over the same problems;
  * holm: Holm-Bonferroni adjustment of a family of p-values.

These are the tests usually recommended for comparing metaheuristics over a
benchmark set (Derrac et al., 2011, Swarm and Evolutionary Computation 1:3-18).
"""

import itertools
import math

from compare_cec2014_results import rank_sum_test  # noqa: F401  (re-exported)


def _average_ranks(values):
    """Ranks 1..n of values, ties getting the average of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + 1 + j) / 2
        i = j
    return ranks


def signed_rank_test(x, y):
    """Two-sided Wilcoxon signed-rank test of paired samples x and y.

    Zero differences are dropped (Wilcoxon's method) and tied absolute
    differences get average ranks. The p-value is exact (by enumerating the
    null distribution of the rank sum) for up to 25 nonzero pairs, and from the
    normal approximation with tie and continuity corrections above that.
    Returns (p-value, W+, W-, number of nonzero pairs); W+ sums the ranks of
    the pairs where x > y.
    """
    differences = [a - b for a, b in zip(x, y) if a != b]
    n = len(differences)
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    ranks = _average_ranks([abs(d) for d in differences])
    w_plus = sum(r for r, d in zip(ranks, differences) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, differences) if d < 0)
    statistic = min(w_plus, w_minus)

    if n <= 25:
        # Ranks doubled so tied (half-integer) ranks stay integers.
        doubled = [int(round(2 * r)) for r in ranks]
        counts = {0: 1}
        for r in doubled:
            nxt = dict(counts)
            for total, ways in counts.items():
                nxt[total + r] = nxt.get(total + r, 0) + ways
            counts = nxt
        target = int(round(2 * statistic))
        tail = sum(ways for total, ways in counts.items() if total <= target)
        p_value = min(1.0, 2 * tail / 2 ** n)
    else:
        mean = n * (n + 1) / 4
        tie_term = 0.0
        for _, group in itertools.groupby(sorted(abs(d) for d in differences)):
            t = len(list(group))
            tie_term += t ** 3 - t
        variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
        z = (abs(statistic - mean) - 0.5) / math.sqrt(variance) if variance > 0 else 0.0
        p_value = math.erfc(max(z, 0.0) / math.sqrt(2))
    return p_value, w_plus, w_minus, n


def _chi2_sf(x, k):
    """Survival function of the chi-squared distribution with k degrees of
    freedom (regularized upper incomplete gamma Q(k/2, x/2))."""
    if x <= 0:
        return 1.0
    a, z = k / 2, x / 2
    if z < a + 1:  # series for the lower incomplete gamma
        term = total = 1.0 / a
        n = 0
        while abs(term) > 1e-15 * abs(total):
            n += 1
            term *= z / (a + n)
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    # continued fraction for the upper incomplete gamma (Lentz's method)
    tiny = 1e-300
    b = z + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, h * math.exp(-z + a * math.log(z) - math.lgamma(a)))


def friedman_test(table):
    """Friedman test of k algorithms (columns) over n problems (rows), lower
    being better. Returns (p-value, average rank of each algorithm). The
    statistic is corrected for ties and compared to chi-squared with k - 1
    degrees of freedom."""
    n, k = len(table), len(table[0])
    rank_rows = [_average_ranks(row) for row in table]
    average = [sum(r[j] for r in rank_rows) / n for j in range(k)]
    if n < 2:
        return float("nan"), average
    sum_squares = sum(r * r for row in rank_rows for r in row)
    totals = [sum(r[j] for r in rank_rows) for j in range(k)]
    spread = sum_squares - n * k * (k + 1) ** 2 / 4
    if spread <= 0:
        return 1.0, average
    chi2 = (k - 1) * sum((t - n * (k + 1) / 2) ** 2 for t in totals) / spread
    return _chi2_sf(chi2, k - 1), average


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for position, i in enumerate(order):
        running = max(running, (m - position) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
