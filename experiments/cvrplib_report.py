#!/usr/bin/env python3
"""Build an HTML report comparing DM-L-SHADE, L-SHADE and HGS-CVRP on CVRPLIB instances.

Reads the results written by cvrplib_collect.py (one folder per protocol under
experiments/results/cvrplib/) and fills experiments/templates/cvrplib_report.html,
borrowing the look of the CEC reports (experiments/templates/report.html). For
every protocol and instance the page shows the gap to the best known value of
each algorithm, the distribution of the final gaps over the runs, convergence
curves (median and interquartile range of the gap at the recording points),
Wilcoxon rank-sum tests between every pair of algorithms and, with two or more
instances, Wilcoxon signed-rank tests paired by instance (Holm-adjusted) and a
Friedman test.

    python3 experiments/cvrplib_report.py
    python3 experiments/cvrplib_report.py --protocols evaluations tmax --output /tmp/cvrp.html
"""

import argparse
import datetime
import itertools
import json
import platform
import re
import statistics
import subprocess
from pathlib import Path

from cvrplib_collect import CANDIDATE, HGS, INSTANCES_DIR, instance_info, read_runs, tmax
from nonparametric import friedman_test, holm, rank_sum_test, signed_rank_test

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = Path(__file__).resolve().parent / "templates"
ORDER = ["L-SHADE", "DM-L-SHADE", HGS]


def quartiles(values):
    if len(values) == 1:
        return [values[0]] * 3
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return [q1, median, q3]


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def instance_report(folder, instance, names, alpha):
    path = INSTANCES_DIR / f"{instance}.vrp"
    customers, best_known = instance_info(path)
    capacity = next((float(m.group(1)) for m in
                     (re.match(r"\s*CAPACITY\s*:\s*([\d.]+)", l) for l in path.read_text().splitlines()) if m), None)
    gap = lambda cost: 100 * (cost - best_known) / best_known
    algorithms = {}
    finals = {}
    for name in names:
        runs = read_runs(folder / name / f"{instance}.txt")
        if not runs:
            continue
        costs = [r[0] for r in runs.values()]
        gaps = [gap(c) for c in costs]
        finals[name] = costs
        curve = None
        length = len(next(iter(runs.values()))[2])
        if length:
            curve = []
            for k in range(length):
                values = [gap(r[2][k]) for r in runs.values() if len(r[2]) > k and r[2][k] == r[2][k]]
                curve.append(quartiles(values) if values else None)
        algorithms[name] = {
            "runs": len(costs),
            "seeds": sorted(runs),
            "costs": [runs[s][0] for s in sorted(runs)],
            "gaps": [gap(runs[s][0]) for s in sorted(runs)],
            "best": min(costs), "worst": max(costs), "median": statistics.median(costs),
            "mean": statistics.mean(costs), "std": statistics.pstdev(costs),
            "bestGap": min(gaps), "meanGap": statistics.mean(gaps), "worstGap": max(gaps),
            "hits": sum(1 for c in costs if c <= best_known),
            "cpu": statistics.mean(r[1] for r in runs.values()),
            "curve": curve,
        }
    pairs = []
    for a, b in itertools.combinations([n for n in ORDER if n in finals], 2):
        p_value, rank_a, rank_b = rank_sum_test(finals[a], finals[b])
        better = a if rank_a < rank_b else b
        pairs.append({"a": a, "b": b, "p": p_value, "better": better if p_value < alpha else None})
    return {"name": instance, "customers": customers, "capacity": capacity, "bks": best_known,
            "tmax": tmax(path), "algorithms": algorithms, "pairs": pairs}


def across_instances(instances, names, alpha):
    complete = [i for i in instances if all(n in i["algorithms"] for n in names)]
    if len(complete) < 2:
        return {"instances": len(complete)}
    table = [[i["algorithms"][n]["meanGap"] for n in names] for i in complete]
    p_friedman, ranks = friedman_test(table) if len(names) > 2 else (None, None)
    others = [n for n in names if n != CANDIDATE]
    tests = [signed_rank_test([i["algorithms"][CANDIDATE]["meanGap"] for i in complete],
                              [i["algorithms"][n]["meanGap"] for i in complete]) for n in others]
    adjusted = holm([t[0] for t in tests])
    return {
        "instances": len(complete),
        "meanGaps": {n: statistics.mean(i["algorithms"][n]["meanGap"] for i in complete) for n in names},
        "friedman": None if p_friedman is None else {"p": p_friedman, "ranks": dict(zip(names, ranks))},
        "signedRank": [{"other": n, "p": t[0], "holm": h, "wPlus": t[1], "wMinus": t[2], "pairs": t[3],
                        "lower": t[2] > t[1]} for n, t, h in zip(others, tests, adjusted)],
        "alpha": alpha,
    }


def protocol_report(folder, alpha):
    settings = json.loads((folder / "protocol.json").read_text())
    names = [n for n in ORDER if (folder / n).is_dir()]
    instance_names = sorted({p.stem for n in names for p in (folder / n).glob("*.txt")},
                            key=lambda n: int(re.search(r"-n(\d+)", n).group(1)))
    instances = [instance_report(folder, i, names, alpha) for i in instance_names]
    return {"key": folder.name, "settings": settings, "names": names, "instances": instances,
            "across": across_instances(instances, names, alpha)}


def hgs_version(hgs_dir):
    try:
        return subprocess.run(["git", "-C", str(hgs_dir), "log", "-1", "--format=%h (%cs)"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def base_style():
    """The fonts and <style> block of the CEC report template."""
    text = (TEMPLATES / "report.html").read_text()
    head = text[:text.index("</style>") + len("</style>")]
    return "\n".join(line for line in head.splitlines() if not line.startswith("<title>"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "experiments" / "results" / "cvrplib",
                        help="folder written by cvrplib_collect.py")
    parser.add_argument("--protocols", nargs="+",
                        help="protocol folders to include (default: all, evaluation budget first)")
    parser.add_argument("--hgs-dir", type=Path, help="HGS-CVRP checkout, to report its version")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level (default: 0.05)")
    parser.add_argument("--output", type=Path, help="HTML file (default: <results>/report.html)")
    args = parser.parse_args()

    folders = [args.results / p for p in args.protocols] if args.protocols else \
        sorted((p.parent for p in args.results.glob("*/protocol.json")), key=lambda p: (p.name != "evaluations", p.name))
    if not folders:
        raise SystemExit(f"no results in {args.results}; run cvrplib_collect.py first")

    data = {
        "generated": datetime.date.today().isoformat(),
        "candidate": CANDIDATE,
        "alpha": args.alpha,
        "machine": {"cpu": cpu_model(), "hgs": hgs_version(args.hgs_dir) if args.hgs_dir else None},
        "protocols": [protocol_report(folder, args.alpha) for folder in folders],
    }
    output = args.output or args.results / "report.html"
    page = (TEMPLATES / "cvrplib_report.html").read_text()
    page = page.replace("/*__BASE_STYLE__*/", base_style())
    page = page.replace("/*__REPORT_DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    output.write_text(page)
    print(f"wrote {output} ({', '.join(p['key'] for p in data['protocols'])})")


if __name__ == "__main__":
    main()
