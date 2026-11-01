#!/usr/bin/env python3
"""Collect L-SHADE / DM-L-SHADE results following the CEC-2026 competition rules.

Implements section 2 of "Evaluation Criteria for CEC 2026 Competition and Special
Session on Bound Constrained Single and Multi-Objective Optimization Considering
both Accuracy and Speed" (TR_BoundConst-MOP-SOP.pdf):

  * the 29 CEC-2017 functions, numbered 1-29, search range [-100, 100]^D, 25
    independent runs per function, MaxFES = 10000*D (all fixed by the solver);
    the competition itself is at D = 30, D = 10 is outside its rules;
  * every run uses the whole budget (there is no early stop at the optimum);
  * the best error (Min_EV) is recorded every 10*D evaluations, and values below
    1e-8 count as zero;
  * one file per algorithm, function and dimension, named
    "AlgorithmName_FunctionNo._D.txt", holding a 1000 x 25 matrix (one row per
    recording point, one column per run). The rules say 1001 rows, the first for
    the initial population, but the organizers' slides say 1000 and every entry's
    file has 1000, which is what the speed comparison needs.

Build the solver first (`make`), then:

    python3 experiments/cec2026_collect.py                # D = 10 and 30
    python3 experiments/cec2026_collect.py --dims 30      # only the competition's dimension

Results are written to experiments/results/cec2026/<AlgorithmName>/. Finished
files are kept, so an interrupted collection resumes where it stopped (use
--force to redo them). The algorithm-complexity measurement (T1, T2) is not
produced by this script.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = {"lshade": "L-SHADE", "dmlshade": "DM-L-SHADE"}
FUNCTIONS = list(range(1, 30))
DIMENSIONS = (10, 30, 50, 100)
CHECKPOINTS = 1000        # number of error values recorded per run
RUNS = 25
ZERO_THRESHOLD = 1e-8     # errors below this are taken as zero


def run_solver(solver, algorithm, function, dim, seed):
    """One run; returns the error at each of the 1000 recording points."""
    command = [
        str(solver),
        "--benchmark", "cec2026",
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
                 f"got {len(errors)} (is the solver up to date? run `make`)")
    return [0.0 if error < ZERO_THRESHOLD else error for error in errors]


def seed_for(base_seed, dim, function, run):
    # Distinct for every (dim, function, run), so no two runs share a seed.
    return base_seed + dim * 1_000_000 + function * 1_000 + run


def result_path(output, name, function, dim):
    return output / name / f"{name}_{function}_{dim}.txt"


def read_matrix(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def is_complete(path, runs):
    """A finished result file is a 1000 x runs matrix."""
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
    rows = [" ".join(f"{run[checkpoint]:.8e}" for run in runs_errors) for checkpoint in range(CHECKPOINTS)]
    path.write_text("\n".join(rows) + "\n")


def collect(args):
    """Run everything that is missing; returns how many result files were written."""
    pending = [(algorithm, dim, function)
               for algorithm in args.algorithms for dim in args.dims for function in args.functions
               if args.force or not is_complete(result_path(args.output, ALGORITHMS[algorithm], function, dim), args.runs)]
    if not pending:
        print("All requested result files already exist (use --force to redo them).")
        return 0

    print(f"{len(pending)} result files to produce, {args.runs} runs each, {args.jobs} runs in parallel", flush=True)
    results = {key: [None] * args.runs for key in pending}
    remaining = {key: args.runs for key in pending}
    written = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        # Submitted file by file so files complete (and are saved) progressively.
        for key in pending:
            algorithm, dim, function = key
            for run in range(args.runs):
                # The same seed for both algorithms, so they start from the same draws.
                seed = seed_for(args.seed, dim, function, run)
                futures[pool.submit(run_solver, args.solver, algorithm, function, dim, seed)] = (key, run)

        try:
            for future in as_completed(futures):
                key, run = futures[future]
                results[key][run] = future.result()
                remaining[key] -= 1
                if remaining[key] == 0:
                    algorithm, dim, function = key
                    name = ALGORITHMS[algorithm]
                    write_matrix(result_path(args.output, name, function, dim), results.pop(key))
                    written += 1
                    print(f"[{written}/{len(pending)}] {name} F{function} D{dim}", flush=True)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"\nInterrupted; {written} files were saved and will be kept.", file=sys.stderr)
            sys.exit(130)
    return written


def write_tables(args):
    """One table per algorithm and dimension, built from the result files on disk."""
    header = f"{'Func.':>5}" + "".join(f"{c:>18}" for c in ("Best", "Worst", "Median", "Mean", "Std"))
    for algorithm in args.algorithms:
        name = ALGORITHMS[algorithm]
        for dim in args.dims:
            lines = []
            for function in args.functions:
                path = result_path(args.output, name, function, dim)
                if is_complete(path, args.runs):
                    final = read_matrix(path)[-1]
                    stats = (min(final), max(final), statistics.median(final),
                             statistics.mean(final), statistics.pstdev(final))
                    lines.append(f"{function:>5}" + "".join(f"{value:>18.8e}" for value in stats))
            if lines:
                table = "\n".join([header] + lines) + "\n"
                (args.output / name / f"{name}_Table_{dim}D.txt").write_text(table)
                print(f"\n{name}, D={dim} ({args.runs} runs, error at MaxFES)")
                print(table, end="")


def main():
    parser = argparse.ArgumentParser(
        description="Collect L-SHADE / DM-L-SHADE results following the CEC-2026 rules.")
    parser.add_argument("--algorithms", nargs="+", choices=sorted(ALGORITHMS),
                        default=list(ALGORITHMS), help="algorithms to run (default: both)")
    parser.add_argument("--dims", nargs="+", type=int, choices=DIMENSIONS, default=[10, 30],
                        help="dimensions (default: 10 30; the competition is at 30)")
    parser.add_argument("--functions", nargs="+", type=int, choices=FUNCTIONS,
                        default=list(FUNCTIONS), help="function numbers (default: 1-29)")
    parser.add_argument("--runs", type=int, default=RUNS,
                        help=f"runs per function; the rules require {RUNS} (default: {RUNS})")
    parser.add_argument("--seed", type=int, default=int(time.time()),
                        help="base seed (default: current time)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="solver processes to run in parallel (default: CPU count)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2026",
                        help="output directory (default: experiments/results/cec2026)")
    parser.add_argument("--solver", type=Path, default=REPO_ROOT / "solver",
                        help="path to the solver binary (default: <repo>/solver)")
    parser.add_argument("--force", action="store_true",
                        help="redo result files that already exist")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not args.solver.exists():
        sys.exit(f"solver not found at {args.solver}; run `make` in {REPO_ROOT} first")
    if args.runs != RUNS:
        print(f"warning: the CEC-2026 rules require {RUNS} runs, using {args.runs}", file=sys.stderr)

    print(f"base seed {args.seed} (pass --seed {args.seed} to reproduce)", flush=True)
    collect(args)
    write_tables(args)


if __name__ == "__main__":
    main()
