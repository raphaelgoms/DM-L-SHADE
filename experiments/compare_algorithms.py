#!/usr/bin/env python3
"""Compare L-SHADE and DM-L-SHADE on one benchmark function.

Runs the solver N times per algorithm (one `./solver ... --runs 1` process per
run, each with its own seed), then reports the mean and standard deviation of
the final error value and of the run time for each algorithm and compares them.

Build the solver first (`make`), then from anywhere:

    python3 experiments/compare_algorithms.py --function 6 --runs 30
"""

import argparse
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ["lshade", "dmlshade"]
LABELS = {"lshade": "L-SHADE", "dmlshade": "DM-L-SHADE"}


TIME_PREFIX = "execution time ="


def run_solver(solver, benchmark, function, algorithm, seed):
    """One solver run; returns (error value, seconds).

    With --runs 1 --show-time the solver prints "execution time = <s> s" (the
    algorithm's run only, excluding process start and data loading) followed
    by a line holding just the error value.
    """
    command = [
        str(solver),
        "--benchmark", benchmark,
        "--f", str(function),
        "--algorithm", algorithm,
        "--runs", "1",
        "--seed", str(seed),
        "--show-time",
    ]
    # The benchmarks load their data from paths relative to the repo root.
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"solver failed ({' '.join(command)}):\n{result.stderr.strip()}")

    error = seconds = None
    try:
        for line in result.stdout.splitlines():
            if line.startswith(TIME_PREFIX):
                seconds = float(line[len(TIME_PREFIX):].strip().removesuffix("s"))
            elif line.strip():
                error = float(line)
    except ValueError:
        error = None
    if error is None or seconds is None:
        sys.exit(f"unexpected solver output for {' '.join(command)}:\n{result.stdout!r}")
    return error, seconds


def collect(solver, benchmark, function, algorithm, runs, base_seed, jobs):
    # Each run needs a distinct seed: the solver otherwise seeds from the wall
    # clock (1 s resolution), so runs started together would be identical.
    # Both algorithms use the same seeds, which makes the experiment repeatable.
    seeds = [base_seed + i for i in range(runs)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(run_solver, solver, benchmark, function, algorithm, seed)
            for seed in seeds
        ]
        return [future.result() for future in futures]


def summarize(values):
    return {
        "mean": statistics.mean(values),
        # Population std (divide by N), matching the solver's own report.
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def print_table(title, summaries, fmt):
    columns = ["mean", "std", "median", "min", "max"]
    print(title)
    print(f"{'algorithm':<12}" + "".join(f"{c:>16}" for c in columns))
    for algorithm in ALGORITHMS:
        row = "".join(f"{summaries[algorithm][c]:>16{fmt}}" for c in columns)
        print(f"{LABELS[algorithm]:<12}{row}")


def print_error_comparison(summaries):
    base = summaries["lshade"]["mean"]
    other = summaries["dmlshade"]["mean"]
    if base == other:
        print("\nBoth algorithms have the same mean error.")
        return

    better, worse = ("dmlshade", "lshade") if other < base else ("lshade", "dmlshade")
    better_mean, worse_mean = summaries[better]["mean"], summaries[worse]["mean"]
    print(f"\n{LABELS[better]} has the lower mean error "
          f"({better_mean:.6e} vs {worse_mean:.6e}", end="")
    if worse_mean > 0:
        print(f", {100 * (worse_mean - better_mean) / worse_mean:.1f}% lower)")
    else:
        print(")")


def print_time_comparison(summaries, jobs):
    base = summaries["lshade"]["mean"]
    other = summaries["dmlshade"]["mean"]
    if base > 0:
        ratio = other / base
        verdict = f"{ratio:.2f}x the mean time of {LABELS['lshade']}"
        print(f"\n{LABELS['dmlshade']} takes {verdict} "
              f"({other:.4f} s vs {base:.4f} s).")
    if jobs > 1:
        print(f"Times were measured with {jobs} runs in parallel and are affected "
              "by CPU contention; use --jobs 1 for cleaner timings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", default="cec2022", choices=["cec2014", "cec2022"],
                        help="benchmark suite (default: cec2022)")
    parser.add_argument("--function", type=int, default=6,
                        help="benchmark function number (default: 6)")
    parser.add_argument("--runs", type=int, default=30,
                        help="runs per algorithm (default: 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the first run; run i uses seed+i (default: 0)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="solver processes to run in parallel (default: CPU count)")
    parser.add_argument("--solver", type=Path, default=REPO_ROOT / "solver",
                        help="path to the solver binary (default: <repo>/solver)")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not args.solver.exists():
        sys.exit(f"solver not found at {args.solver}; run `make` in {REPO_ROOT} first")

    print(f"{args.benchmark} function {args.function}, {args.runs} runs per algorithm\n")

    results = {}
    for algorithm in ALGORITHMS:
        results[algorithm] = collect(args.solver, args.benchmark, args.function,
                                     algorithm, args.runs, args.seed, args.jobs)

    errors = {a: summarize([error for error, _ in runs]) for a, runs in results.items()}
    times = {a: summarize([seconds for _, seconds in runs]) for a, runs in results.items()}

    print_table("Error value", errors, ".6e")
    print_error_comparison(errors)

    print()
    print_table("Time per run (seconds)", times, ".4f")
    print_time_comparison(times, args.jobs)


if __name__ == "__main__":
    main()
