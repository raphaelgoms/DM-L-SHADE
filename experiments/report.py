#!/usr/bin/env python3
"""Build an HTML report of the L-SHADE / DM-L-SHADE results of a CEC suite.

Reads the result files written by cec2014_collect.py, cec2017_collect.py or
cec2022_collect.py and fills experiments/templates/report.html with them,
producing one self-contained page with:

  * the per-dimension results tables of the suite's results template (best,
    worst, median, mean and std of the error at the end of the run);
  * the per-function comparison of the matching compare_cec20XX_results.py
    (Wilcoxon rank-sum test) and, where the rules define one, the suite's
    score (CEC-2017 and CEC-2022 section 2.3; CEC-2014 has none);
  * convergence curves (median and interquartile range of the error at the
    suite's recording points) for every function.

Dimensions are picked up from whatever result files exist, so the report can
be rebuilt as more of them are collected:

    python3 experiments/report.py --benchmark cec2017
    python3 experiments/report.py --benchmark cec2022 --participants <CEC2022 download>/Results_and_Ranking/Comparison

The report is written to experiments/results/<suite>/report.html by default.
"""

import argparse
import datetime
import json
import math
import re
import ctypes
import ctypes.util
import io
import statistics
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from compare_cec2014_results import rank_sum_test

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "templates" / "report.html"
BASELINE, CANDIDATE = "L-SHADE", "DM-L-SHADE"
ALGORITHMS = (BASELINE, CANDIDATE)
TARGET_ERROR = 1e-8
FRACTIONS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PARAMETERS = (
    "Both algorithms use the same parameters on every function and dimension: initial population "
    "18·D, memory size 6, archive rate 2.6, p-best rate 0.11. DM-L-SHADE adds an elite set of 10% of "
    "the population, mined every 168 generations into a number of clusters equal to 14.68% of the "
    "elite size (K-means)."
)


def fraction_checkpoints(dim, maxfes):
    """CEC-2014 and CEC-2017: (0.01, 0.02, 0.03, 0.05, 0.1, ..., 1.0) * MaxFES."""
    return [round(fraction * maxfes) for fraction in FRACTIONS]


def cec2022_checkpoints(dim, maxfes):
    """CEC-2022: floor(D^(k/5-3) * MaxFES), k = 0..15 (as computed by the solver)."""
    return [math.floor(dim ** (k / 5 - 3) * maxfes + 1e-6) for k in range(16)]


SUITES = {
    "cec2014": {
        "name": "CEC 2014",
        "subtitle": "Single-objective real-parameter numerical optimization",
        "functions": list(range(1, 31)),
        "functionsLabel": "30 (F1–F30)",
        "dims": [10, 30, 50, 100],
        "runs": 51,
        "maxfes": lambda dim: 10000 * dim,
        "checkpoints": fraction_checkpoints,
        "feTerm": False,
        "stopsAtTarget": False,
        "optimum": {f: 100 * f for f in range(1, 31)},
        "names": {
            1: "Rotated high conditioned elliptic", 2: "Rotated Bent Cigar", 3: "Rotated Discus",
            4: "Rosenbrock", 5: "Ackley", 6: "Weierstrass", 7: "Griewank", 8: "Shifted Rastrigin",
            9: "Rastrigin", 10: "Shifted Schwefel", 11: "Schwefel", 12: "Katsuura", 13: "HappyCat",
            14: "HGBat", 15: "Expanded Griewank plus Rosenbrock", 16: "Expanded Schaffer F6",
            17: "Hybrid 1 (N=3)", 18: "Hybrid 2 (N=3)", 19: "Hybrid 3 (N=4)", 20: "Hybrid 4 (N=4)",
            21: "Hybrid 5 (N=5)", 22: "Hybrid 6 (N=5)",
            23: "Composition 1 (N=5)", 24: "Composition 2 (N=3)", 25: "Composition 3 (N=3)",
            26: "Composition 4 (N=5)", 27: "Composition 5 (N=5)", 28: "Composition 6 (N=5)",
            29: "Composition 7 (N=3)", 30: "Composition 8 (N=3)",
        },
        "groups": [
            ("Unimodal", [1, 2, 3]),
            ("Simple multimodal", list(range(4, 17))),
            ("Hybrid", list(range(17, 23))),
            ("Composition", list(range(23, 31))),
        ],
        "tableNumbers": {10: "III", 30: "IV", 50: "V", 100: "VI"},
        "complexity": "Table VII, algorithm complexity (T0, T1, T̂2 on F18 at 10D, 30D and 50D, section 3)",
        "lede": "collected under the competition rules: 51 runs per function and MaxFES = 10000·D, "
                "with every run using its whole budget.",
        "setting": [
            "30 functions, search range [−100, 100]<sup>D</sup>, uniform random initialization.",
            "51 independent runs per function and dimension; the seed of each run is derived from a "
            "time-based base seed.",
            "MaxFES = 10000·D. Every run uses the whole budget.",
            "The error F(x) − F(x*) is recorded at 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, …, 1.0 × MaxFES. "
            "Errors below 10<sup>−8</sup> count as zero.",
            PARAMETERS,
        ],
        "notes": [
            "F*<sub>i</sub> = 100·i. Functions 1–3 are rotated only and functions 8 and 10 shifted only; "
            "every other basic function is shifted and rotated.",
            "The comparison uses a two-sided Wilcoxon rank-sum test on the final errors. Because it "
            "compares raw errors, it also reacts to how close each run gets to the bottom of the same "
            "local optimum.",
            "The CEC 2014 rules define no score, so this report has none.",
        ],
        "testText": "two-sided Wilcoxon rank-sum test on the final errors",
    },
    "cec2017": {
        "name": "CEC 2017",
        "subtitle": "Single-objective bound-constrained optimization",
        "functions": [1] + list(range(3, 31)),
        "functionsLabel": "29 (F1, F3–F30)",
        "dims": [10, 30, 50, 100],
        "runs": 51,
        "maxfes": lambda dim: 10000 * dim,
        "checkpoints": fraction_checkpoints,
        "feTerm": False,
        "stopsAtTarget": True,
        "optimum": {f: 100 * f for f in range(1, 31)},
        "names": {
            1: "Bent Cigar", 3: "Zakharov", 4: "Rosenbrock", 5: "Rastrigin", 6: "Expanded Schaffer F6",
            7: "Lunacek bi-Rastrigin", 8: "Non-continuous Rastrigin", 9: "Levy", 10: "Schwefel",
            11: "Hybrid 1 (N=3)", 12: "Hybrid 2 (N=3)", 13: "Hybrid 3 (N=3)", 14: "Hybrid 4 (N=4)",
            15: "Hybrid 5 (N=4)", 16: "Hybrid 6 (N=4)", 17: "Hybrid 7 (N=5)", 18: "Hybrid 8 (N=5)",
            19: "Hybrid 9 (N=5)", 20: "Hybrid 10 (N=6)",
            21: "Composition 1 (N=3)", 22: "Composition 2 (N=3)", 23: "Composition 3 (N=4)",
            24: "Composition 4 (N=4)", 25: "Composition 5 (N=5)", 26: "Composition 6 (N=5)",
            27: "Composition 7 (N=6)", 28: "Composition 8 (N=6)", 29: "Composition 9 (N=3)",
            30: "Composition 10 (N=3)",
        },
        "groups": [
            ("Unimodal", [1, 3]),
            ("Simple multimodal", list(range(4, 11))),
            ("Hybrid", list(range(11, 21))),
            ("Composition", list(range(21, 31))),
        ],
        "tableNumbers": {10: "III", 30: "IV", 50: "V", 100: "VI"},
        "complexity": "Table VII, algorithm complexity (T0, T1, T̂2 on F18 at 10D, 30D and 50D, section 3)",
        "lede": "collected under the competition rules: 51 runs per function, MaxFES = 10000·D, "
                "and a run stops once its error falls below 10<sup>−8</sup>.",
        "setting": [
            "29 functions, search range [−100, 100]<sup>D</sup>, uniform random initialization.",
            "51 independent runs per function and dimension; the seed of each run is derived from a "
            "time-based base seed.",
            "MaxFES = 10000·D. A run ends at MaxFES or as soon as its error drops below 10<sup>−8</sup>.",
            "The error F(x) − F(x*) is recorded at 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, …, 1.0 × MaxFES; "
            "recording points after an early stop hold the final error. Errors below 10<sup>−8</sup> "
            "count as zero.",
            PARAMETERS,
        ],
        "notes": [
            "Functions keep the numbering of the official code, 1 and 3–30, with F*<sub>i</sub> = 100·i. "
            "F2 was removed from the suite, so the definitions document numbers the same functions 1–29.",
            "The comparison uses a two-sided Wilcoxon rank-sum test on the final errors. Because it "
            "compares raw errors, it also reacts to how close each run gets to the bottom of the same "
            "local optimum.",
            "The score of section 2.3 is defined relative to the best of the compared algorithms. With "
            "two algorithms, the better one gets 100 by construction.",
        ],
        "testText": "two-sided Wilcoxon rank-sum test on the final errors",
    },
    "cec2022": {
        "name": "CEC 2022",
        "subtitle": "Single-objective bound-constrained numerical optimization",
        "functions": list(range(1, 13)),
        "functionsLabel": "12 (F1–F12)",
        "dims": [10, 20],
        "runs": 30,
        "maxfes": lambda dim: 1000000 if dim == 20 else 200000,
        "checkpoints": cec2022_checkpoints,
        "feTerm": True,
        "stopsAtTarget": True,
        "optimum": {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800, 7: 2000, 8: 2200,
                    9: 2300, 10: 2400, 11: 2600, 12: 2700},
        "names": {
            1: "Zakharov", 2: "Rosenbrock", 3: "Expanded Schaffer f6", 4: "Non-continuous Rastrigin",
            5: "Levy", 6: "Hybrid 1 (N=3)", 7: "Hybrid 2 (N=6)", 8: "Hybrid 3 (N=5)",
            9: "Composition 1 (N=5)", 10: "Composition 2 (N=4)", 11: "Composition 3 (N=5)",
            12: "Composition 4 (N=6)",
        },
        "groups": [
            ("Unimodal", [1]),
            ("Basic", [2, 3, 4, 5]),
            ("Hybrid", [6, 7, 8]),
            ("Composition", [9, 10, 11, 12]),
        ],
        "tableNumbers": {10: "II", 20: "III"},
        "complexity": "Table IV, algorithm complexity (T0, T1, T̂2 on F1 at 10D and 20D)",
        "lede": "collected under the competition rules: 30 runs per function with the official seeds, "
                "MaxFES = 200,000 (10D) or 1,000,000 (20D), and a run stops once its error reaches "
                "10<sup>−8</sup>.",
        "setting": [
            "12 functions, search range [−100, 100]<sup>D</sup>, uniform random initialization.",
            "30 independent runs per function and dimension, seeded from <code>Rand_Seeds.txt</code> "
            "as the rules prescribe.",
            "MaxFES = 200,000 for 10D and 1,000,000 for 20D. A run ends at MaxFES or as soon as its "
            "error reaches 10<sup>−8</sup>.",
            "The error F(x) − F(x*) is recorded after ⌊D<sup>k/5−3</sup>·MaxFES⌋ evaluations, "
            "k = 0, …, 15, with 10<sup>−8</sup> recorded once a run has reached it, plus the "
            "evaluations used (FE<sub>term</sub>).",
            PARAMETERS,
        ],
        "notes": [
            "The rules reward speed as well as accuracy: trials are ordered by final error and, among "
            "those that reached 10<sup>−8</sup>, by the evaluations they needed. The Wilcoxon test and "
            "the score both use that ordering.",
            "An algorithm's score on a function is the sum of the ranks of its trials minus n(n+1)/2. "
            "With two algorithms this is the number of the rival's trials each trial beats (ties count "
            "half), so the two scores on a function add up to 30 × 30 = 900.",
            "Errors at the 10<sup>−8</sup> target are stored as 10<sup>−8</sup>, so a function every "
            "run solves shows a mean of 1.00E-08.",
        ],
        "testText": "two-sided Wilcoxon rank-sum test on the trials, ordered by final error and then by FE<sub>term</sub>",
    },
}
ALIASES = {"cec14": "cec2014", "cec17": "cec2017", "cec22": "cec2022"}


def read_matrix(results, name, function, dim):
    path = results / name / f"{name}_{function}_{dim}.txt"
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def available_dims(results, suite):
    pattern = re.compile(rf"^{re.escape(BASELINE)}_\d+_(\d+)\.txt$")
    dims = {int(m.group(1)) for f in (results / BASELINE).glob("*.txt") if (m := pattern.match(f.name))}
    return sorted(dims & set(suite["dims"]))


def available_functions(results, suite, dim):
    return [f for f in suite["functions"]
            if all((results / name / f"{name}_{f}_{dim}.txt").exists() for name in ALGORITHMS)]


def quartiles(values):
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return [q1, median, q3]


def algorithm_stats(error_rows, fe_terms):
    final = error_rows[-1]
    stats = {
        "best": min(final),
        "worst": max(final),
        "median": statistics.median(final),
        "mean": statistics.mean(final),
        "std": statistics.pstdev(final),  # same convention as the collect scripts
        "solved": sum(1 for v in final if v <= TARGET_ERROR),
        "curve": [quartiles(row) for row in error_rows],
    }
    if fe_terms is not None:
        stats["meanFes"] = statistics.mean(fe_terms)
    return stats


def u_score(x, y):
    """Trials of y beaten by trials of x, ties counting half (CEC-2022 score)."""
    return sum((a < b) + 0.5 * (a == b) for a in x for b in y)


def dimension_report(results, suite, dim, alpha):
    checkpoints = suite["checkpoints"](dim, suite["maxfes"](dim))
    functions = []
    runs = None
    for function in available_functions(results, suite, dim):
        error_rows, samples, stats = {}, {}, {}
        for name in ALGORITHMS:
            matrix = read_matrix(results, name, function, dim)
            rows = matrix[:len(checkpoints)]
            fe_terms = [int(v) for v in matrix[len(checkpoints)]] if suite["feTerm"] else None
            error_rows[name] = rows
            # CEC-2022 orders trials by final error, then by evaluations used.
            samples[name] = list(zip(rows[-1], fe_terms)) if fe_terms else rows[-1]
            stats[name] = algorithm_stats(rows, fe_terms)
        runs = len(error_rows[BASELINE][-1])

        p_value, rank_base, rank_cand = rank_sum_test(samples[BASELINE], samples[CANDIDATE])
        symbol = ("+" if rank_cand < rank_base else "-") if p_value < alpha else "="
        entry = {"f": function, "p": p_value, "symbol": symbol, "stats": stats}
        if suite["feTerm"]:
            entry["scores"] = {BASELINE: u_score(samples[BASELINE], samples[CANDIDATE]),
                               CANDIDATE: u_score(samples[CANDIDATE], samples[BASELINE])}
        functions.append(entry)

    report = {"dim": dim, "maxfes": suite["maxfes"](dim), "checkpoints": checkpoints,
              "runs": runs, "functions": functions}
    if suite["feTerm"] and functions:
        report["score"] = cec2022_score(report)
    return report


def cec2022_score(report):
    totals = {name: sum(e["scores"][name] for e in report["functions"]) for name in ALGORITHMS}
    n_functions = len(report["functions"])
    per_function = report["runs"] ** 2
    return {
        "label": f"CEC 2022 score at D = {report['dim']} (section 2.3)",
        "values": totals,
        "max": per_function * n_functions,
        "digits": 1,
        "note": f"Sum over {n_functions} functions of the trials each algorithm beats. The two scores "
                f"add up to {per_function * n_functions:,} ({report['runs']} × {report['runs']} "
                f"per function).",
    }


def cec2017_score(dimensions):
    """Section 2.3 of the CEC-2017 rules, as in compare_cec2017_results.py."""
    weights = {10: 0.1, 30: 0.2, 50: 0.3, 100: 0.4}
    se = {name: 0.0 for name in ALGORITHMS}
    sr = {name: 0.0 for name in ALGORITHMS}
    for report in dimensions:
        weight = weights[report["dim"]]
        for entry in report["functions"]:
            mean_base = entry["stats"][BASELINE]["mean"]
            mean_cand = entry["stats"][CANDIDATE]["mean"]
            se[BASELINE] += weight * mean_base
            se[CANDIDATE] += weight * mean_cand
            ranks = (1.0, 2.0) if mean_base < mean_cand else (2.0, 1.0) if mean_base > mean_cand else (1.5, 1.5)
            sr[BASELINE] += weight * ranks[0]
            sr[CANDIDATE] += weight * ranks[1]

    def part(total, minimum):
        return 50.0 if total == 0 else (1 - (total - minimum) / total) * 50

    score1 = {name: part(se[name], min(se.values())) for name in ALGORITHMS}
    score2 = {name: part(sr[name], min(sr.values())) for name in ALGORITHMS}
    dims = ", ".join(f"{report['dim']}D" for report in dimensions)
    return {
        "label": "CEC 2017 score (section 2.3, out of 100)",
        "values": {name: score1[name] + score2[name] for name in ALGORITHMS},
        "max": 100,
        "digits": 2,
        "note": f"Score 1 (sum of mean errors): {score1[BASELINE]:.2f} vs {score1[CANDIDATE]:.2f}. "
                f"Score 2 (ranks): {score2[BASELINE]:.2f} vs {score2[CANDIDATE]:.2f}. "
                f"Computed over {dims}; the rules weight 10D, 30D, 50D and 100D by 0.1, 0.2, 0.3 and 0.4.",
    }


def read_xlsx(path):
    """Cell values of every sheet of an .xlsx file, as lists of rows of strings
    (None for empty cells). Only what the organizers' spreadsheets need."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{ns}t")) for si in root.findall(f"{ns}si")]
        sheet_names = sorted((n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
                             key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
        sheets = []
        for name in sheet_names:
            rows = []
            for row in ET.fromstring(archive.read(name)).iter(f"{ns}row"):
                cells = {}
                for cell in row.findall(f"{ns}c"):
                    letters = re.match(r"[A-Z]+", cell.get("r")).group(0)
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter) - 64
                    value = cell.find(f"{ns}v")
                    if value is not None:
                        cells[column - 1] = shared[int(value.text)] if cell.get("t") == "s" else value.text
                rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
            sheets.append(rows)
    return sheets


def head_to_head(a, b):
    """Points of a against b on one function, as ranking.m counts them: every
    trial of a meets every trial of b; the lower error wins, equal errors go to
    fewer evaluations, and a full tie is half a point each. a and b are lists
    of (error, FE_term)."""
    points = 0.0
    for error_a, fes_a in a:
        for error_b, fes_b in b:
            if error_a < error_b or (error_a == error_b and fes_a < fes_b):
                points += 1
            elif error_a == error_b and fes_a == fes_b:
                points += 0.5
    return points


def cec2022_ranking(participants_dir, results, dimensions):
    """Final ranking of the CEC-2022 competition with L-SHADE and DM-L-SHADE
    added, recomputed from the organizers' Results_and_Ranking/Comparison
    spreadsheets (final error and FE_term of every trial of the 13 entries)."""
    ranking_book = read_xlsx(participants_dir / "Ranking.xlsx")
    official_matrix, summary = ranking_book[0], ranking_book[1]
    titles = {row[2]: row[3].strip() for row in ranking_book[2][1:] if len(row) > 3} if len(ranking_book) > 2 else {}
    entries = [{"name": row[1], "official": int(row[5]), "officialTotal": float(row[4]),
                "title": titles.get(row[1], ""), "ours": False} for row in summary[1:] if row and row[1]]
    names = [e["name"] for e in entries]

    dims = [report["dim"] for report in dimensions if report["dim"] in (10, 20)]
    trials = {}  # trials[name][dim][function index] = [(error, FE_term), ...]
    for name in names:
        trials[name] = {}
        for dim in dims:
            errors, fes = read_xlsx(participants_dir / f"{name}_{dim}.xlsx")[:2]
            trials[name][dim] = [[(float(errors[run][p]), float(fes[run][p])) for run in range(len(errors))]
                                 for p in range(12)]
    for name in ALGORITHMS:
        trials[name] = {}
        for dim in dims:
            per_function = []
            for function in range(1, 13):
                matrix = read_matrix(results, name, function, dim)
                # The entries record runs that reached 1e-8 as 0; ours store 1e-8.
                errors = [0.0 if v <= TARGET_ERROR else v for v in matrix[15]]
                per_function.append(list(zip(errors, matrix[16])))
            trials[name][dim] = per_function
        entries.append({"name": name, "official": None, "officialTotal": None,
                        "title": "This work" if name == CANDIDATE else "Baseline (Tanabe and Fukunaga, 2014)",
                        "ours": True})

    everyone = [e["name"] for e in entries]
    points = {dim: {a: {b: 0.0 for b in everyone if b != a} for a in everyone} for dim in dims}
    for dim in dims:
        for i, a in enumerate(everyone):
            for b in everyone[i + 1:]:
                for p in range(12):
                    points_a = head_to_head(trials[a][dim][p], trials[b][dim][p])
                    points[dim][a][b] += points_a
                    points[dim][b][a] += len(trials[a][dim][p]) * len(trials[b][dim][p]) - points_a

    # Check the recomputation against the organizers' matrix. ranking.m builds
    # variable names as 'S' + str(k) + str(l), so pairs such as (1, 11) and
    # (11, 1) share a name and one overwrites the other.
    official = {}
    dim = None
    for row in official_matrix:
        if row and row[0] and row[0].endswith("-D"):
            dim, header = int(row[0][:-2]), row[1:1 + len(names)]
            official[dim] = {}
        elif dim and row and row[0] in names:
            official[dim][row[0]] = {header[j]: float(row[1 + j]) for j in range(len(names)) if header[j] != row[0]}
    labels = {}
    for k in range(1, len(names) + 1):
        for l in range(1, len(names) + 1):
            if k != l:
                labels.setdefault(f"{k}{l}", []).append((names[k - 1], names[l - 1]))
    colliding = {pair for group in labels.values() if len(group) > 1 for pair in group}
    checked = mismatched = 0
    unexplained = []
    mismatched_pairs = set()
    for dim in dims:
        for a in names:
            for b in names:
                if a == b or a not in official.get(dim, {}):
                    continue
                checked += 1
                if abs(points[dim][a][b] - official[dim][a][b]) > 1e-6:
                    mismatched += 1
                    mismatched_pairs.add(tuple(sorted((a, b))))
                    if (a, b) not in colliding:
                        unexplained.append(f"{a} vs {b} at {dim}D")
    note = f"Recomputing the organizers' comparison reproduces {checked - mismatched} of {checked} cells of the official matrix of head-to-head points."
    if mismatched and not unexplained:
        pairs = ", ".join(f"{a} vs {b}" for a, b in sorted(mismatched_pairs))
        note += (f" The other {mismatched} are the cells of {pairs}, which ranking.m overwrites: it names "
                 f"its variables 'S' + k + l, so pairs such as (1, 11) and (11, 1) get the same name. "
                 f"The ranking below uses the recomputed points.")
    elif unexplained:
        note += f" Cells that differ for no known reason: {', '.join(unexplained)}."

    return {
        "dims": dims,
        "entries": entries,
        "points": points,
        "perPair": 12 * 30 * 30,
        "note": note,
    }


def read_rar(data):
    """Members of a RAR archive as (name, bytes), through the system libarchive
    (there is no RAR reader in the standard library). None if it is missing."""
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("archive") or "libarchive.so.13")
    except OSError:
        return None
    lib.archive_read_new.restype = ctypes.c_void_p
    for function in ("archive_read_support_format_all", "archive_read_support_filter_all", "archive_read_free"):
        getattr(lib, function).argtypes = [ctypes.c_void_p]
    lib.archive_read_open_memory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    lib.archive_entry_pathname.restype = ctypes.c_char_p
    lib.archive_entry_size.argtypes = [ctypes.c_void_p]
    lib.archive_entry_size.restype = ctypes.c_int64
    lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.archive_read_data.restype = ctypes.c_ssize_t

    archive = lib.archive_read_new()
    lib.archive_read_support_format_all(archive)
    lib.archive_read_support_filter_all(archive)
    members = []
    if lib.archive_read_open_memory(archive, data, len(data)) == 0:
        entry = ctypes.c_void_p()
        while lib.archive_read_next_header(archive, ctypes.byref(entry)) == 0:
            name = lib.archive_entry_pathname(entry).decode(errors="replace")
            size = lib.archive_entry_size(entry)
            if size > 0:
                buffer = ctypes.create_string_buffer(size)
                read = lib.archive_read_data(archive, buffer, size)
                members.append((name, buffer.raw[:max(read, 0)]))
    lib.archive_read_free(archive)
    return members


def archive_files(name, data, skipped):
    """Every file inside an archive, opening nested zip, tar.gz and rar archives
    too, as (path, bytes). Archives that cannot be read are added to skipped."""
    lower = name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [(n, archive.read(n)) for n in archive.namelist() if not n.endswith("/")]
    elif lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers() if m.isfile()]
    elif lower.endswith(".rar"):
        members = read_rar(data)
        if members is None:
            skipped.append(name)
            return
    else:
        yield name, data
        return
    for member, content in members:
        yield from archive_files(f"{name}/{member}", content, skipped)


def matrix_from_text(text):
    """Rows of numbers of a whitespace- or comma-separated results file. The
    MOS entries write CSV with a header row and a label column, both dropped."""
    rows = []
    for line in text.splitlines():
        cells = line.split(",") if "," in line else line.split()
        values = []
        for cell in cells:
            try:
                values.append(float(cell))
            except ValueError:
                pass  # header or label field
        if values:
            rows.append(values)
    return rows


# The CEC-2017 entries, as listed in the organizers' comparison slides
# (Bound-Constrained-Comparisons.pdf), with the official scores of those slides
# and how to find each entry's files in "Results for all papers.zip". `pattern`
# matches a file name and gives (function, dimension), or (dimension, function)
# when `dim_first`; LSHADE-cnEpSin only sent statistics (best, worst, median,
# mean, std per function), which are enough for the score but not for tests.
CEC2017_ENTRIES = [
    ("EBOwithCMAR", "Improving the local search capability of Effective Butterfly Optimizer using Covariance Matrix Adapted Retreat phase", 50.0, 48.012, r"EBOwithCMAR_(\d+)_(\d+)\.dat$", False),
    ("jSO", "Single Objective Real-Parameter Optimization Algorithm jSO", 49.689, 47.076, r"jSO_(\d+)_(\d+)\.txt$", False),
    ("LSHADE-cnEpSin", "Ensemble Sinusoidal Differential Covariance Matrix Adaptation with Euclidean Neighborhood for Solving CEC2017 Benchmark Problems", 46.816, 49.743, r"results_stat_(\d+)\.txt$", None),
    ("LSHADE_SPACMA", "LSHADE with Semi-Parameter Adaptation Hybrid with CMA-ES for Solving CEC 2017 Benchmark Problems", 46.438, 50.0, r"LSHADE_SPACMA_(\d+)_(\d+)\.txt$", False),
    ("DES", "A Differential Evolution Strategy", 45.935, 43.202, r"DES_(\d+)_(\d+)\.txt$", False),
    ("MM_OED", "Multi-method based Orthogonal Experimental Design Algorithm for Solving CEC2017 Competition Problems", 45.956, 40.116, r"MM_OED_(\d+)_(\d+)\.txt$", False),
    ("IDEbestNsize", "Enhanced Individual-dependent Differential Evolution with Population Size Adaptation", 29.850, 27.679, r"IDEbestNsize_(\d+)_(\d+)\.txt$", True),
    ("RB-IPOP-CMA-ES", "A Version of IPOP-CMA-ES Algorithm with Midpoint for CEC 2017 Single Objective Bound Constrained Problems", 3.7925, 33.612, r"RB-IPOP-CMA-ES_(\d+)_(\d+)\.txt$", False),
    ("MOS-SOCO2013", "A comparison of three large-scale global optimizers on the CEC 2017 single objective real parameter numerical optimization benchmark", 18.942, 17.343, r"MOS-CEC2013_(\d+)_(\d+)\.csv$", False),
    ("MOS-SOCO2011", "A comparison of three large-scale global optimizers on the CEC 2017 single objective real parameter numerical optimization benchmark", 11.085, 19.305, r"MOS-SOCO2011_(\d+)_(\d+)\.csv$", False),
    ("PPSO", "Proactive Particles in Swarm Optimization: a Settings-Free Algorithm for Real-Parameter Single Objective Optimization Problems", 3.9254, 17.355, r"PPSO_(\d+)_(\d+)\.txt$", False),
    ("DYYPO", "Dynamic Yin-Yang Pair Optimization and its Performance on Single Objective Real Parameter Problems of CEC 2017", 0.59337, 17.031, r"DYYPO_(\d+)_(\d+)\.txt$", False),
    ("TLBO-FL", "Teaching Learning Based Optimization with Focused Learning and its Performance on CEC2017 functions", 0.028731, 16.246, r"TLBO-FL_(\d+)_(\d+)\.txt$", False),
]
CEC2017_WEIGHTS = {10: 0.1, 30: 0.2, 50: 0.3, 100: 0.4}


def cec2017_section23(means, names, dims, functions):
    """Score1, Score2 (section 2.3) of each algorithm from mean errors
    means[name][dim][function], ranks with ties averaged."""
    se = {a: sum(CEC2017_WEIGHTS[d] * means[a][d][f] for d in dims for f in functions) for a in names}
    sr = {a: 0.0 for a in names}
    for d in dims:
        for f in functions:
            ordered = sorted((means[a][d][f], a) for a in names)
            i = 0
            while i < len(ordered):
                j = i
                while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                    j += 1
                for k in range(i, j):
                    sr[ordered[k][1]] += CEC2017_WEIGHTS[d] * (i + 1 + j) / 2
                i = j

    def part(total, minimum):
        return 50.0 if total == 0 else (1 - (total - minimum) / total) * 50

    return ({a: part(se[a], min(se.values())) for a in names},
            {a: part(sr[a], min(sr.values())) for a in names})


def cec2017_ranking(archive_path, results, dimensions, alpha):
    """The CEC-2017 ranking (section 2.3 score) with L-SHADE and DM-L-SHADE
    added, from the entries' result files in the organizers' archive
    (Codes-of-Top-Methods-and-results.zip or the "Results for all papers.zip"
    inside it), plus Wilcoxon tests of DM-L-SHADE against each entry."""
    data = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        inner = [n for n in outer.namelist() if n.endswith("Results for all papers.zip")]
        if inner:
            data = outer.read(inner[0])
    skipped = []
    files = list(archive_files("Results for all papers.zip", data, skipped))

    functions = SUITES["cec2017"]["functions"]
    all_dims = SUITES["cec2017"]["dims"]
    our_dims = [report["dim"] for report in dimensions]
    finals = {}  # finals[name][dim][function] = final errors of the runs (None: statistics only)
    means = {}
    missing = []
    for name, _, _, _, pattern, dim_first in CEC2017_ENTRIES:
        finals[name] = {d: {} for d in all_dims}
        means[name] = {d: {} for d in all_dims}
        for path, content in files:
            match = re.search(pattern, path.rsplit("/", 1)[-1])
            if not match:
                continue
            rows = matrix_from_text(content.decode(errors="replace"))
            if dim_first is None:  # statistics: one row per function 1-30, mean in column 4
                dim = int(match.group(1))
                for function, row in enumerate(rows, start=1):
                    if function in functions and dim in means[name]:
                        means[name][dim][function] = row[3]
                continue
            a, b = int(match.group(1)), int(match.group(2))
            function, dim = (b, a) if dim_first else (a, b)
            if function in functions and dim in finals[name]:
                final = [0.0 if v < TARGET_ERROR else v for v in rows[-1]]
                finals[name][dim][function] = final
                means[name][dim][function] = statistics.mean(final)
        if any(f not in means[name][d] for d in all_dims for f in functions):
            missing.append(name)
    names = [entry[0] for entry in CEC2017_ENTRIES if entry[0] not in missing]

    # Our algorithms, in the dimensions collected so far.
    for name in ALGORITHMS:
        finals[name] = {}
        means[name] = {}
        for d in our_dims:
            finals[name][d] = {f: read_matrix(results, name, f, d)[-1] for f in functions}
            means[name][d] = {f: statistics.mean(finals[name][d][f]) for f in functions}

    # Check the entries' data against the official slides (all four dimensions).
    score1, score2 = cec2017_section23(means, names, all_dims, functions)
    official = {entry[0]: (entry[2], entry[3]) for entry in CEC2017_ENTRIES}
    worst1 = max(abs(score1[a] - official[a][0]) for a in names)
    worst2 = max(abs(score2[a] - official[a][1]) for a in names)

    wilcoxon = {}
    for name in names:
        wilcoxon[name] = {}
        for d in our_dims:
            if not finals[name][d]:
                continue
            counts = {"+": 0, "=": 0, "-": 0}
            for f in functions:
                p_value, rank_ours, rank_theirs = rank_sum_test(finals[CANDIDATE][d][f], finals[name][d][f])
                counts[("+" if rank_ours < rank_theirs else "-") if p_value < alpha else "="] += 1
            wilcoxon[name][d] = counts
    wilcoxon[BASELINE] = {}
    for d in our_dims:
        counts = {"+": 0, "=": 0, "-": 0}
        for entry in next(r for r in dimensions if r["dim"] == d)["functions"]:
            counts[entry["symbol"]] += 1
        wilcoxon[BASELINE][d] = counts

    entries = [{"name": e[0], "title": e[1], "officialScore": e[2] + e[3], "ours": False,
                "statsOnly": e[5] is None} for e in CEC2017_ENTRIES if e[0] in names]
    entries += [{"name": CANDIDATE, "title": "This work", "officialScore": None, "ours": True, "statsOnly": False},
                {"name": BASELINE, "title": "Baseline (Tanabe and Fukunaga, 2014)", "officialScore": None,
                 "ours": True, "statsOnly": False}]

    note = (f"Recomputing section 2.3 over all four dimensions from the entries' files reproduces the "
            f"official Score 1 of every entry (largest difference {worst1:.3f})")
    note += (f" and their Score 2 to within {worst2:.2f} points." if worst2 < 0.01 else
             f", but not Score 2, which differs by up to {worst2:.1f} points with ranks averaged over ties "
             f"(no tie rule or rounding of the mean errors tried reproduces it). Score 2 depends on how each "
             f"entry ranks on every function, so a difference in the files the organizers used, or in how "
             f"they ranked, is the likely cause. The official column shows the published scores.")
    if missing:
        note += f" Left out because their files are incomplete or unreadable: {', '.join(missing)}."
    if skipped:
        note += f" Archives that could not be opened: {', '.join(skipped)}."
    return {
        "kind": "score",
        "dims": our_dims,
        "weights": {d: CEC2017_WEIGHTS[d] for d in our_dims},
        "functions": functions,
        "entries": entries,
        "means": {name: {d: means[name][d] for d in our_dims} for name in names + list(ALGORITHMS)},
        "wilcoxon": wilcoxon,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True, choices=sorted(SUITES) + sorted(ALIASES),
                        help="suite whose results to report")
    parser.add_argument("--results", type=Path,
                        help="directory written by the collect script (default: experiments/results/<suite>)")
    parser.add_argument("--output", type=Path,
                        help="HTML file to write (default: <results>/report.html)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level of the Wilcoxon test (default: 0.05)")
    parser.add_argument("--participants", type=Path,
                        help="CEC-2022 only: the organizers' Results_and_Ranking/Comparison folder, to add "
                             "the competition ranking with the 13 entries of 2022")
    args = parser.parse_args()

    key = ALIASES.get(args.benchmark, args.benchmark)
    suite = SUITES[key]
    results = args.results or REPO_ROOT / "experiments" / "results" / key

    dimensions = [report for report in (dimension_report(results, suite, dim, args.alpha)
                                        for dim in available_dims(results, suite))
                  if report["functions"]]
    if not dimensions:
        raise SystemExit(f"no result files found in {results}; run {key}_collect.py first")

    data = {
        "suite": {k: v for k, v in suite.items() if not callable(v) and k not in ("checkpoints", "maxfes")},
        "benchmark": key,
        "generated": datetime.date.today().isoformat(),
        "algorithms": list(ALGORITHMS),
        "alpha": args.alpha,
        "dimensions": dimensions,
        "score": cec2017_score(dimensions) if key == "cec2017" else None,
        "ranking": None,
    }
    if args.participants:
        if key == "cec2022":
            data["ranking"] = cec2022_ranking(args.participants, results, dimensions)
        elif key == "cec2017":
            data["ranking"] = cec2017_ranking(args.participants, results, dimensions, args.alpha)
        else:
            parser.error("--participants is only available for cec2017 and cec2022")
    elif key == "cec2022":
        print("note: pass --participants <Results_and_Ranking/Comparison> to include the ranking "
              "against the CEC-2022 entries")
    elif key == "cec2017":
        print("note: pass --participants <Codes-of-Top-Methods-and-results.zip> to include the ranking "
              "against the CEC-2017 entries")

    output = args.output or results / "report.html"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text()
    page = page.replace("__REPORT_TITLE__", f"DM-L-SHADE on {suite['name']}")
    page = page.replace("/*__REPORT_DATA__*/null", payload)
    output.write_text(page)
    collected = ", ".join(f"D={report['dim']}" for report in dimensions)
    print(f"wrote {output} ({collected})")


if __name__ == "__main__":
    main()
