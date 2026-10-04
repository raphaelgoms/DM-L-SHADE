#!/usr/bin/env python3
"""Collect L-SHADE / DM-L-SHADE results on CVRPLIB instances, optionally with HGS-CVRP.

Runs the solver (`--benchmark cvrplib`) on instances of data/cvrplib/X (the X
set of Uchoa et al., 2017) with the same seeds for every algorithm, and can
run HGS-CVRP (Vidal, 2022; https://github.com/vidalt/HGS-CVRP) on the same
instances and seeds as the reference. Two protocols:

  * evaluations (default): the solver runs for 10000*n evaluations (n
    customers), recording the best cost at 0.01, 0.02, 0.03, 0.05, 0.1, ...,
    1.0 x MaxFES. HGS-CVRP stops by its own criterion (20,000 iterations
    without improvement) unless --hgs-time gives it a CPU time limit in
    seconds, or `match`, the mean CPU time DM-L-SHADE took on the instance;
  * CPU time (--cpu-time tmax or --cpu-time SECONDS): the protocol of Vidal
    (2022) for the X set. Every algorithm, HGS-CVRP included, gets the same CPU
    time, Tmax = n x 240/100 seconds with `tmax`, and records the best cost at
    1, 2, 5, 10, 15, 20, 30, 50, 75 and 100% of it. Run with --jobs at most the
    number of physical cores, so runs do not share a core.

    python3 experiments/cvrplib_collect.py --instances X-n101-k25 --runs 10 --hgs <hgs>
    python3 experiments/cvrplib_collect.py --instances all --max-customers 200 \\
        --cpu-time tmax --hgs <hgs> --jobs 8

Results go to experiments/results/cvrplib/<protocol>/<Algorithm>/<instance>.txt
(<protocol> being "evaluations", "tmax" or "cpu<seconds>s"), one line per run:
"seed cost cpu_seconds" followed by the checkpoint costs. A protocol.json in
each folder records the settings, so runs of different protocols never mix.
Runs already on disk are skipped, so an interrupted collection resumes where
it stopped.

The summary printed at the end gives, per instance, the best and mean gap to
the best known value (the .sol next to each instance) and the mean CPU time,
with a Wilcoxon rank-sum test of DM-L-SHADE against each algorithm over the
runs; over the instances, the mean gaps, Wilcoxon signed-rank tests of
DM-L-SHADE against each algorithm paired by instance (Holm-adjusted) and a
Friedman test of all of them. --summary-only prints it without running.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nonparametric import friedman_test, holm, rank_sum_test, signed_rank_test

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = REPO_ROOT / "data" / "cvrplib" / "X"
SOLVER_ALGORITHMS = {"lshade": "L-SHADE", "dmlshade": "DM-L-SHADE"}
HGS = "HGS"
CANDIDATE = "DM-L-SHADE"
EVALUATION_FRACTIONS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TIME_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]


class TimedPopen(subprocess.Popen):
    """Popen that records the child's user + system CPU time when reaped."""
    cpu_seconds = 0.0

    def _try_wait(self, wait_flags):  # same contract as subprocess.Popen._try_wait
        try:
            pid, status, usage = os.wait4(self.pid, wait_flags)
        except ChildProcessError:
            return self.pid, 0
        if pid == self.pid:
            self.cpu_seconds = usage.ru_utime + usage.ru_stime
        return pid, status


def run_cpu_timed(command, cwd=None):
    with TimedPopen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode, process.cpu_seconds


def instance_info(path):
    """(number of customers, best known value or None) of a .vrp file."""
    dimension = None
    for line in path.read_text().splitlines():
        match = re.match(r"\s*DIMENSION\s*:\s*(\d+)", line)
        if match:
            dimension = int(match.group(1))
            break
    best = None
    solution = path.with_suffix(".sol")
    if solution.exists():
        for line in solution.read_text().splitlines():
            if line.startswith("Cost"):
                best = float(line.split()[1])
    return dimension - 1, best


def tmax(path):
    """Time limit of Vidal (2022): n x 240/100 seconds for n customers."""
    return instance_info(path)[0] * 240 / 100


def run_solver(solver, algorithm, instance, seed, cpu_time):
    command = [str(solver), "--benchmark", "cvrplib", "--instance", str(instance), "--algorithm", algorithm,
               "--runs", "1", "--seed", str(seed), "--checkpoints"]
    if cpu_time:
        command += ["--cpu-time", f"{cpu_time:.3f}"]
    expected = len(TIME_FRACTIONS if cpu_time else EVALUATION_FRACTIONS)
    stdout, stderr, code, cpu = run_cpu_timed(command, cwd=REPO_ROOT)
    if code != 0:
        raise RuntimeError(f"solver failed ({' '.join(command)}):\n{stderr.strip()}")
    costs = [float(v) for v in stdout.split()]
    if len(costs) != expected:
        raise RuntimeError(f"expected {expected} checkpoint costs from {' '.join(command)}, got {stdout!r} "
                           f"(is the solver up to date? run `make`)")
    return costs[-1], cpu, costs


TRACE = re.compile(r"T\(s\)\s+([\d.]+)\s+\|\s+Feas\s+\d+\s+([\d.]+)")


def run_hgs(hgs, instance, seed, time_limit, record):
    """One HGS-CVRP run. With record, the best feasible cost at TIME_FRACTIONS
    of time_limit is read from its progress trace (CPU seconds and best
    feasible cost of the current population every 100 iterations; with a time
    limit HGS-CVRP restarts, so the running minimum is taken)."""
    with tempfile.TemporaryDirectory() as tmp:
        solution = Path(tmp) / "hgs.sol"
        command = [str(hgs), str(instance), str(solution), "-seed", str(seed)]
        if time_limit:
            command += ["-t", f"{time_limit:.3f}"]
        command += ["-log", "1", "-nbIterTraces", "100"] if record else ["-log", "0"]
        stdout, stderr, code, cpu = run_cpu_timed(command)
        if code != 0 or not solution.exists():
            raise RuntimeError(f"HGS failed ({' '.join(command)}):\n{stderr.strip() or stdout.strip()[-500:]}")
        cost = next(float(l.split()[1]) for l in solution.read_text().splitlines() if l.startswith("Cost"))
    checkpoints = []
    if record:
        trace = [(float(t), float(c)) for t, c in TRACE.findall(stdout)]
        best = float("inf")
        points = iter(trace)
        current = next(points, None)
        for fraction in TIME_FRACTIONS:
            limit = fraction * time_limit
            while current is not None and current[0] <= limit:
                best = min(best, current[1])
                current = next(points, None)
            checkpoints.append(best if best < float("inf") else float("nan"))
        checkpoints[-1] = min(checkpoints[-1], cost) if checkpoints[-1] == checkpoints[-1] else cost
    return cost, cpu, checkpoints


def result_path(output, name, instance):
    return output / name / f"{instance}.txt"


def read_runs(path):
    """{seed: (cost, cpu_seconds, checkpoint costs)} of a results file."""
    runs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            values = line.split()
            if len(values) >= 3:
                runs[int(values[0])] = (float(values[1]), float(values[2]), [float(v) for v in values[3:]])
    return runs


def collect(jobs, args, output, describe):
    """Runs the (name, instance, seed, function) jobs in parallel, appending
    each finished run to its results file right away."""
    if not jobs:
        return
    print(f"{describe}: {len(jobs)} runs, {args.jobs} in parallel")
    locks = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(function): (name, instance, seed) for name, instance, seed, function in jobs}
        try:
            for future in as_completed(futures):
                name, instance, seed = futures[future]
                cost, cpu, checkpoints = future.result()
                path = result_path(output, name, instance)
                with locks.setdefault(path, threading.Lock()):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a") as out:
                        out.write(" ".join([str(seed), f"{cost:.0f}", f"{cpu:.3f}"] +
                                           [f"{c:.0f}" if c == c else "nan" for c in checkpoints]) + "\n")
                done += 1
                print(f"[{done}/{len(jobs)}] {name} {instance} seed {seed}: cost {cost:.0f}, {cpu:.1f} s CPU")
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"\nInterrupted; the {done} finished runs were saved and will be kept.", file=sys.stderr)
            sys.exit(130)
        except RuntimeError as error:
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(str(error))


def gap(cost, best_known):
    return 100 * (cost - best_known) / best_known


def summarize(args, output, instances, names):
    print(f"\nGap to the best known value (%) and mean CPU time per run; per instance, Wilcoxon rank-sum "
          f"test of {CANDIDATE} against each algorithm over the runs, alpha = {args.alpha} "
          f"(+ better, - worse, = no difference)")
    header = f"{'Instance':<14}{'n':>5}{'BKS':>9}"
    for name in names:
        header += f"  | {name:<10} {'best':>7} {'mean':>7} {'CPU s':>8}"
    print(header)
    mean_gaps = {}  # mean_gaps[instance][name]
    counts = {name: {"+": 0, "=": 0, "-": 0} for name in names if name != CANDIDATE}
    for instance in instances:
        customers, best_known = instance_info(INSTANCES_DIR / f"{instance}.vrp")
        runs = {name: read_runs(result_path(output, name, instance)) for name in names}
        if not any(runs.values()) or best_known is None:
            continue
        line = f"{instance:<14}{customers:>5}{best_known:>9.0f}"
        for name in names:
            costs = [r[0] for r in runs[name].values()]
            if not costs:
                line += f"  | {'':<10} {'-':>7} {'-':>7} {'-':>8}"
                continue
            mark = ""
            if name != CANDIDATE and runs[CANDIDATE]:
                p_value, rank_ours, rank_theirs = rank_sum_test([r[0] for r in runs[CANDIDATE].values()], costs)
                mark = ("+" if rank_ours < rank_theirs else "-") if p_value < args.alpha else "="
                counts[name][mark] += 1
            mean_gaps.setdefault(instance, {})[name] = gap(statistics.mean(costs), best_known)
            cpu = statistics.mean(r[1] for r in runs[name].values())
            label = f"{len(costs)} runs {mark}".strip()
            line += f"  | {label:<10} {gap(min(costs), best_known):>7.2f} " \
                    f"{gap(statistics.mean(costs), best_known):>7.2f} {cpu:>8.1f}"
        print(line)

    complete = [i for i in mean_gaps if all(n in mean_gaps[i] for n in names)]
    print(f"\nOver the {len(complete)} instances with results for every algorithm:")
    for name in names:
        if complete:
            gaps = [mean_gaps[i][name] for i in complete]
            print(f"  {name}: mean gap {statistics.mean(gaps):.2f}% (min {min(gaps):.2f}%, max {max(gaps):.2f}%)")
    for name, count in counts.items():
        print(f"  per instance, {CANDIDATE} vs {name}: {count['+']} better, {count['-']} worse, "
              f"{count['=']} no significant difference")
    if len(complete) < 2:
        print("  (tests over instances need at least two instances)")
        return
    others = [n for n in names if n != CANDIDATE]
    tests = [signed_rank_test([mean_gaps[i][CANDIDATE] for i in complete], [mean_gaps[i][n] for i in complete])
             for n in others]
    adjusted = holm([t[0] for t in tests])
    for name, (p_value, w_plus, w_minus, pairs), p_holm in zip(others, tests, adjusted):
        verdict = "lower" if w_minus > w_plus else "higher"
        print(f"  Wilcoxon signed-rank, {CANDIDATE} vs {name} (mean gap per instance): p = {p_value:.4f}, "
              f"Holm p = {p_holm:.4f}; {CANDIDATE}'s gaps are {verdict} on {pairs} differing instances")
    if len(names) > 2:
        p_value, ranks = friedman_test([[mean_gaps[i][n] for n in names] for i in complete])
        print(f"  Friedman test of all algorithms: p = {p_value:.4f}; average ranks "
              + ", ".join(f"{n} {r:.2f}" for n, r in zip(names, ranks)))


def main():
    parser = argparse.ArgumentParser(
        description="Collect L-SHADE / DM-L-SHADE (and optionally HGS-CVRP) results on CVRPLIB instances.")
    parser.add_argument("--instances", nargs="+", required=True,
                        help="instance names in data/cvrplib/X (e.g. X-n101-k25), or 'all'")
    parser.add_argument("--max-customers", type=int,
                        help="with 'all', only instances with at most this many customers")
    parser.add_argument("--algorithms", nargs="+", choices=sorted(SOLVER_ALGORITHMS),
                        default=list(SOLVER_ALGORITHMS), help="solver algorithms to run (default: both)")
    parser.add_argument("--runs", type=int, default=10, help="runs per instance and algorithm (default: 10)")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the first run; run k uses seed + k, the same for every algorithm (default: 1)")
    parser.add_argument("--cpu-time", default=None,
                        help="'tmax' (n x 240/100 s, Vidal 2022) or a number of seconds: CPU-time protocol for "
                             "every algorithm (default: the evaluation budget)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="runs in parallel (default: CPU count; with --cpu-time use at most the physical cores)")
    parser.add_argument("--hgs", type=Path, help="path to the HGS-CVRP executable, to run it as well")
    parser.add_argument("--hgs-time", default=None,
                        help="evaluation protocol only: CPU time limit of HGS-CVRP in seconds, or 'match' for "
                             "DM-L-SHADE's mean CPU time on each instance (default: its own stopping criterion)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level of the summary tests")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "results" / "cvrplib",
                        help="results directory; each protocol gets a subfolder (default: experiments/results/cvrplib)")
    parser.add_argument("--solver", type=Path, default=REPO_ROOT / "solver",
                        help="path to the solver binary (default: <repo>/solver)")
    parser.add_argument("--summary-only", action="store_true", help="print the summary of the runs on disk")
    args = parser.parse_args()

    if args.instances == ["all"]:
        instances = sorted((p.stem for p in INSTANCES_DIR.glob("*.vrp")),
                           key=lambda n: int(re.search(r"-n(\d+)", n).group(1)))
        if args.max_customers:
            instances = [n for n in instances if instance_info(INSTANCES_DIR / f"{n}.vrp")[0] <= args.max_customers]
    else:
        instances = args.instances
    missing = [n for n in instances if not (INSTANCES_DIR / f"{n}.vrp").exists()]
    if missing:
        parser.error(f"not found in {INSTANCES_DIR}: {', '.join(missing)}")

    if args.cpu_time is None:
        protocol = "evaluations"
        cpu_time_of = lambda path: None
    elif args.cpu_time == "tmax":
        protocol = "tmax"
        cpu_time_of = tmax
    else:
        try:
            seconds = float(args.cpu_time)
        except ValueError:
            parser.error("--cpu-time must be 'tmax' or a number of seconds")
        protocol = f"cpu{seconds:g}s"
        cpu_time_of = lambda path: seconds
    if args.cpu_time is not None and args.hgs_time is not None:
        parser.error("--hgs-time only applies to the evaluation protocol; with --cpu-time HGS-CVRP gets the same time")
    if args.hgs_time not in (None, "match"):
        try:
            float(args.hgs_time)
        except ValueError:
            parser.error("--hgs-time must be a number of seconds or 'match'")
    if args.hgs_time == "match" and "dmlshade" not in args.algorithms:
        parser.error("--hgs-time match needs DM-L-SHADE in --algorithms")

    output = args.output / protocol
    settings = {"protocol": protocol, "cpu_time": args.cpu_time, "hgs_time": args.hgs_time,
                "checkpoint_fractions": TIME_FRACTIONS if args.cpu_time else EVALUATION_FRACTIONS}
    settings_path = output / "protocol.json"
    if settings_path.exists() and not args.summary_only:
        previous = json.loads(settings_path.read_text())
        if previous.get("hgs_time") != settings["hgs_time"] and args.hgs:
            sys.exit(f"{output} holds HGS-CVRP runs with --hgs-time {previous.get('hgs_time')}; "
                     f"use another --output to change it")

    names = [SOLVER_ALGORITHMS[a] for a in sorted(args.algorithms, key=list(SOLVER_ALGORITHMS).index)]
    if args.hgs or (output / HGS).exists():
        names.append(HGS)
    seeds = [args.seed + k for k in range(args.runs)]

    if not args.summary_only:
        if not args.solver.exists():
            sys.exit(f"solver not found at {args.solver}; run `make` in {REPO_ROOT} first")
        if args.hgs and not args.hgs.exists():
            sys.exit(f"HGS-CVRP not found at {args.hgs}")
        output.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        jobs = []
        for instance in instances:
            path = INSTANCES_DIR / f"{instance}.vrp"
            for algorithm in args.algorithms:
                name = SOLVER_ALGORITHMS[algorithm]
                done = read_runs(result_path(output, name, instance))
                jobs += [(name, instance, seed,
                          lambda a=algorithm, p=path, s=seed: run_solver(args.solver, a, p, s, cpu_time_of(p)))
                         for seed in seeds if seed not in done]
        collect(jobs, args, output, "solver")

        if args.hgs:
            jobs = []
            for instance in instances:
                path = INSTANCES_DIR / f"{instance}.vrp"
                limit = cpu_time_of(path)
                if args.hgs_time == "match":
                    ours = read_runs(result_path(output, CANDIDATE, instance))
                    limit = statistics.mean(r[1] for s, r in ours.items() if s in seeds)
                elif args.hgs_time:
                    limit = float(args.hgs_time)
                record = args.cpu_time is not None
                done = read_runs(result_path(output, HGS, instance))
                jobs += [(HGS, instance, seed,
                          lambda p=path, s=seed, t=limit: run_hgs(args.hgs, p, s, t, record))
                         for seed in seeds if seed not in done]
            collect(jobs, args, output, "HGS-CVRP")

    summarize(args, output, instances, names)


if __name__ == "__main__":
    main()
