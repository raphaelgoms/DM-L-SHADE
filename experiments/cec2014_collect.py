#!/usr/bin/env python3
"""Collect L-SHADE / DM-L-SHADE results following the CEC-2014 competition rules.

Implements section 2 of "Problem Definitions and Evaluation Criteria for the
CEC 2014 Special Session and Competition on Single Objective Real-Parameter
Numerical Optimization" (Liang, Qu, Suganthan):

  * 30 functions, D in {10, 30, 50, 100}, 51 independent runs per function,
    MaxFES = 10000*D, search range [-100, 100]^D (all fixed by the solver);
  * the error value is recorded at (0.01, 0.02, 0.03, 0.05, 0.1, ..., 0.9, 1.0)
    * MaxFES in every run, and error values below 1e-8 count as zero;
  * one file per algorithm, function and dimension, named
    "AlgorithmName_FunctionNo._D.txt", holding a 14 x 51 matrix (one row per
    checkpoint, one column per run);
  * for each dimension, a table with the best, worst, median, mean and
    standard deviation of the error at MaxFES over the 51 runs.

Build the solver first (`make`), then:

    python3 experiments/cec2014_collect.py --dims 10 30            # initial submission
    python3 experiments/cec2014_collect.py --dims 10 30 50 100     # final version

Results are written to experiments/results/cec2014/<AlgorithmName>/. Finished
files are kept, so an interrupted collection resumes where it stopped (use
--force to redo them). The rules do not cover the algorithm-complexity
measurement (T0, T1, T2), which this script does not produce.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = {"lshade": "L-SHADE", "dmlshade": "DM-L-SHADE"}
FUNCTIONS = range(1, 31)
DIMENSIONS = (10, 30, 50, 100)
CHECKPOINTS = 14          # number of error values recorded per run
ZERO_THRESHOLD = 1e-8     # errors below this are taken as zero


def run_solver(solver, algorithm, function, dim, seed):
    """One run; returns the error at each of the 14 checkpoints."""
    command = [
        str(solver),
        "--benchmark", "cec2014",
        "--f", str(function),
        "--dim", str(dim),
        "--algorithm", algorithm,
        "--runs", "1",
        "--seed", str(seed),
        "--checkpoints",
    ]
    # The benchmarks load their data from paths relative to the repo root.
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"solver failed ({' '.join(command)}):\n{result.stderr.strip()}")
    try:
        errors = [float(value) for value in result.stdout.split()]
    except ValueError:
        errors = []
    if len(errors) != CHECKPOINTS:
        sys.exit(f"expected {CHECKPOINTS} checkpoint errors from {' '.join(command)}, "
                 f"got:\n{result.stdout!r}\n(is the solver up to date? run `make`)")
    return [0.0 if error < ZERO_THRESHOLD else error for error in errors]


def seed_for(base_seed, dim, function, run):
    # Distinct for every (dim, function, run), so no two runs share a seed.
    return base_seed + dim * 1_000_000 + function * 1_000 + run


def result_path(output, name, function, dim):
    return output / name / f"{name}_{function}_{dim}.txt"


def read_matrix(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def is_complete(path, runs):
    """A finished result file is a 14 x runs matrix."""
    if not path.exists():
        return False
    try:
        matrix = read_matrix(path)
    except ValueError:
        return False
    return len(matrix) == CHECKPOINTS and all(len(row) == runs for row in matrix)


def write_matrix(path, runs_errors):
    """runs_errors[run][checkpoint] -> file with one row per checkpoint, one column per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in range(CHECKPOINTS):
        rows.append(" ".join(f"{run[checkpoint]:.15e}" for run in runs_errors))
    path.write_text("\n".join(rows) + "\n")


def collect(args, name_of):
    """Run everything that is missing; returns how many result files were written."""
    pending = []  # (algorithm, dim, function)
    for algorithm in args.algorithms:
        for dim in args.dims:
            for function in args.functions:
                path = result_path(args.output, name_of[algorithm], function, dim)
                if args.force or not is_complete(path, args.runs):
                    pending.append((algorithm, dim, function))

    if not pending:
        print("All requested result files already exist (use --force to redo them).")
        return 0

    print(f"{len(pending)} result files to produce, {args.runs} runs each, "
          f"{args.jobs} runs in parallel")

    results = {key: [None] * args.runs for key in pending}
    remaining = {key: args.runs for key in pending}
    written = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        # Submitted file by file so files complete (and are saved) progressively.
        for key in pending:
            algorithm, dim, function = key
            for run in range(args.runs):
                seed = seed_for(args.seed, dim, function, run)
                future = pool.submit(run_solver, args.solver, algorithm, function, dim, seed)
                futures[future] = (key, run)

        try:
            for future in as_completed(futures):
                key, run = futures[future]
                results[key][run] = future.result()
                remaining[key] -= 1
                if remaining[key] == 0:
                    algorithm, dim, function = key
                    name = name_of[algorithm]
                    write_matrix(result_path(args.output, name, function, dim), results.pop(key))
                    written += 1
                    print(f"[{written}/{len(pending)}] {name} F{function} D{dim}")
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"\nInterrupted; {written} files were saved and will be kept.", file=sys.stderr)
            sys.exit(130)

    return written


def summarize(path):
    """Statistics of the error at MaxFES (last checkpoint row) over the runs."""
    final_errors = read_matrix(path)[-1]
    return (
        min(final_errors),
        max(final_errors),
        statistics.median(final_errors),
        statistics.mean(final_errors),
        # Population std (divide by N), like the solver's own report; the
        # rules only say "standard variance".
        statistics.pstdev(final_errors),
    )


def write_tables(args, name_of):
    """One table per algorithm and dimension, built from the result files on disk."""
    header = f"{'Func.':>5}" + "".join(f"{c:>18}" for c in ("Best", "Worst", "Median", "Mean", "Std"))
    for algorithm in args.algorithms:
        name = name_of[algorithm]
        for dim in args.dims:
            lines = []
            for function in args.functions:
                path = result_path(args.output, name, function, dim)
                if not is_complete(path, args.runs):
                    continue
                stats = summarize(path)
                lines.append(f"{function:>5}" + "".join(f"{value:>18.8e}" for value in stats))
            if not lines:
                continue
            table = "\n".join([header] + lines) + "\n"
            (args.output / name / f"{name}_Table_{dim}D.txt").write_text(table)
            print(f"\n{name}, D={dim} ({args.runs} runs, error at MaxFES)")
            print(table, end="")


def write_zips(args, name_of):
    """Zip each algorithm's result files, as the rules ask for submission."""
    for algorithm in args.algorithms:
        name = name_of[algorithm]
        archive = args.output / f"{name}.zip"
        files = sorted(
            result_path(args.output, name, function, dim)
            for dim in args.dims for function in args.functions
            if is_complete(result_path(args.output, name, function, dim), args.runs)
        )
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                zip_file.write(file, arcname=file.name)
        print(f"wrote {archive} ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Collect L-SHADE / DM-L-SHADE results following the CEC-2014 rules.")
    parser.add_argument("--algorithms", nargs="+", choices=sorted(ALGORITHMS),
                        default=list(ALGORITHMS), help="algorithms to run (default: both)")
    parser.add_argument("--dims", nargs="+", type=int, choices=DIMENSIONS, default=[10, 30],
                        help="dimensions (default: 10 30; the final version needs 10 30 50 100)")
    parser.add_argument("--functions", nargs="+", type=int, choices=FUNCTIONS,
                        default=list(FUNCTIONS), help="function numbers (default: 1-30)")
    parser.add_argument("--runs", type=int, default=51,
                        help="runs per function; the rules require 51 (default: 51)")
    parser.add_argument("--seed", type=int, default=int(time.time()),
                        help="base seed (default: current time, as the rules require)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="solver processes to run in parallel (default: CPU count)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2014",
                        help="output directory (default: experiments/results/cec2014)")
    parser.add_argument("--solver", type=Path, default=REPO_ROOT / "solver",
                        help="path to the solver binary (default: <repo>/solver)")
    parser.add_argument("--force", action="store_true",
                        help="redo result files that already exist")
    parser.add_argument("--zip", action="store_true",
                        help="also zip each algorithm's result files")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not args.solver.exists():
        sys.exit(f"solver not found at {args.solver}; run `make` in {REPO_ROOT} first")
    if args.runs != 51:
        print(f"warning: the CEC-2014 rules require 51 runs, using {args.runs}", file=sys.stderr)

    print(f"base seed {args.seed} (pass --seed {args.seed} to reproduce)")
    collect(args, ALGORITHMS)
    write_tables(args, ALGORITHMS)
    if args.zip:
        write_zips(args, ALGORITHMS)


if __name__ == "__main__":
    main()
