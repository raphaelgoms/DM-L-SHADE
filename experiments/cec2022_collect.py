#!/usr/bin/env python3
"""Collect L-SHADE / DM-L-SHADE results following the CEC-2022 competition rules.

Implements section 2 of "Problem Definitions and Evaluation Criteria for the
CEC 2022 Special Session and Competition on Single Objective Bound Constrained
Numerical Optimization" (Kumar, Price, Mohamed, Hadi, Suganthan):

  * 12 functions, D in {10, 20}, 30 independent runs per function,
    MaxFES = 200,000 (D = 10) or 1,000,000 (D = 20), search range
    [-100, 100]^D (all fixed by the solver);
  * a run terminates at MaxFES or as soon as the error drops below 1e-8
    (also done by the solver);
  * the seed of each run comes from data/cec22/Rand_Seeds.txt, indexed by
    seed_ind = mod((D/10 * func_no * Runs + run_id) - Runs, 1000) + 1;
  * the error value is recorded after floor(D^(k/5 - 3) * MaxFES) evaluations,
    k = 0, ..., 15, with 1e-8 recorded once a run has reached it, and the
    number of evaluations at termination (FE_term) as the 17th entry;
  * one file per algorithm, function and dimension, named
    "AlgorithmName_FunctionNo._D.txt", holding a 17 x 30 matrix (one row per
    record, one column per run);
  * for each dimension, a table with the best, worst, median, mean and
    standard deviation of the error at MaxFES over the 30 runs.

Build the solver first (`make`), then:

    python3 experiments/cec2022_collect.py
    python3 experiments/cec2022_collect.py --dims 10 --functions 1 2 3 --zip

Results are written to experiments/results/cec2022/<AlgorithmName>/. Finished
files are kept, so an interrupted collection resumes where it stopped (use
--force to redo them). The algorithm-complexity measurement (T0, T1, T2) is
not produced by this script.
"""

import argparse
import os
import statistics
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = {"lshade": "L-SHADE", "dmlshade": "DM-L-SHADE"}
FUNCTIONS = range(1, 13)
DIMENSIONS = (10, 20)
CHECKPOINTS = 16          # error values recorded per run, followed by FE_term
ROWS = CHECKPOINTS + 1
TARGET_ERROR = 1e-8       # a run that reaches this error terminates
SEEDS_FILE = REPO_ROOT / "data" / "cec22" / "Rand_Seeds.txt"


def run_solver(solver, algorithm, function, dim, seed):
    """One run; returns its 17 records (16 errors, then FE_term)."""
    command = [
        str(solver),
        "--benchmark", "cec2022",
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
        values = [float(value) for value in result.stdout.split()]
    except ValueError:
        values = []
    if len(values) != ROWS:
        sys.exit(f"expected {ROWS} values (errors and FE_term) from {' '.join(command)}, "
                 f"got:\n{result.stdout!r}\n(is the solver up to date? run `make`)")
    # The solver reports errors below 1e-8 as 0; the rules record them as 1e-8.
    errors = [max(error, TARGET_ERROR) for error in values[:CHECKPOINTS]]
    return errors + [int(values[CHECKPOINTS])]


def load_seeds():
    if not SEEDS_FILE.exists():
        sys.exit(f"{SEEDS_FILE} not found; it ships with the CEC-2022 code (input_data/Rand_Seeds.txt)")
    return [int(float(value)) for value in SEEDS_FILE.read_text().split()]


def official_seed(seeds, dim, function, run, runs):
    """Seed prescribed by the rules for run `run` (0-based) of a function."""
    run_id = run + 1
    seed_ind = (dim // 10 * function * runs + run_id) - runs
    return seeds[seed_ind % 1000]  # mod(seed_ind, 1000) + 1, as a 0-based index


def result_path(output, name, function, dim):
    return output / name / f"{name}_{function}_{dim}.txt"


def read_matrix(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def is_complete(path, runs):
    """A finished result file is a 17 x runs matrix."""
    if not path.exists():
        return False
    try:
        matrix = read_matrix(path)
    except ValueError:
        return False
    return len(matrix) == ROWS and all(len(row) == runs for row in matrix)


def write_matrix(path, runs_records):
    """runs_records[run][row] -> file with one row per record, one column per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in range(CHECKPOINTS):
        rows.append(" ".join(f"{run[checkpoint]:.15e}" for run in runs_records))
    rows.append(" ".join(str(run[CHECKPOINTS]) for run in runs_records))
    path.write_text("\n".join(rows) + "\n")


def collect(args, name_of, seed_of):
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
                seed = seed_of(dim, function, run)
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
    """Statistics of the error at MaxFES (last error row) over the runs."""
    final_errors = read_matrix(path)[CHECKPOINTS - 1]
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
        description="Collect L-SHADE / DM-L-SHADE results following the CEC-2022 rules.")
    parser.add_argument("--algorithms", nargs="+", choices=sorted(ALGORITHMS),
                        default=list(ALGORITHMS), help="algorithms to run (default: both)")
    parser.add_argument("--dims", nargs="+", type=int, choices=DIMENSIONS, default=list(DIMENSIONS),
                        help="dimensions (default: 10 20)")
    parser.add_argument("--functions", nargs="+", type=int, choices=FUNCTIONS,
                        default=list(FUNCTIONS), help="function numbers (default: 1-12)")
    parser.add_argument("--runs", type=int, default=30,
                        help="runs per function; the rules require 30 (default: 30)")
    parser.add_argument("--seed", type=int,
                        help="base seed to use instead of the official seeds of Rand_Seeds.txt")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="solver processes to run in parallel (default: CPU count)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "results" / "cec2022",
                        help="output directory (default: experiments/results/cec2022)")
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
    if args.runs != 30:
        print(f"warning: the CEC-2022 rules require 30 runs, using {args.runs}", file=sys.stderr)

    if args.seed is None:
        seeds = load_seeds()
        def seed_of(dim, function, run):
            return official_seed(seeds, dim, function, run, args.runs)
        print(f"using the official seeds of {SEEDS_FILE.relative_to(REPO_ROOT)}")
    else:
        def seed_of(dim, function, run):
            # Distinct for every (dim, function, run), so no two runs share a seed.
            return args.seed + dim * 1_000_000 + function * 1_000 + run
        print(f"base seed {args.seed} (not the official seeds)")

    collect(args, ALGORITHMS, seed_of)
    write_tables(args, ALGORITHMS)
    if args.zip:
        write_zips(args, ALGORITHMS)


if __name__ == "__main__":
    main()
