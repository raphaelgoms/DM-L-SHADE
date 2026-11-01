# Experiments

Python scripts that run the solver to collect results, compare L-SHADE with DM-L-SHADE and build HTML reports. They use only the standard library (no numpy, scipy or openpyxl).

Build the solver first, from the repository root:

```
make
```

The scripts call `./solver` themselves (one process per run, in parallel with `--jobs`) and can be launched from any directory. Each script documents its rules and options in its docstring and in `--help`.

## Workflow

Every benchmark follows the same three steps: **collect** the runs, **compare** the algorithms in the terminal, and build an HTML **report**.

| Benchmark | Collect | Compare | Report |
| --- | --- | --- | --- |
| CEC-2014 | [cec2014_collect.py](cec2014_collect.py) | [compare_cec2014_results.py](compare_cec2014_results.py) | [report.py](report.py) `--benchmark cec2014` |
| CEC-2017 | [cec2017_collect.py](cec2017_collect.py) | [compare_cec2017_results.py](compare_cec2017_results.py) | [report.py](report.py) `--benchmark cec2017` |
| CEC-2022 | [cec2022_collect.py](cec2022_collect.py) | [compare_cec2022_results.py](compare_cec2022_results.py) | [report.py](report.py) `--benchmark cec2022` |
| CEC-2026 | [cec2026_collect.py](cec2026_collect.py) | [compare_cec2026_results.py](compare_cec2026_results.py) | [report.py](report.py) `--benchmark cec2026` |
| CVRPLIB | [cvrplib_collect.py](cvrplib_collect.py) | `cvrplib_collect.py --summary-only` | [cvrplib_report.py](cvrplib_report.py) |

Results go to `results/<suite>/<AlgorithmName>/`. The collect scripts keep files already on disk, so an interrupted collection resumes where it stopped (`--force` redoes them in the CEC scripts).

## CEC benchmark suites

### Collecting: `cec20XX_collect.py`

Run L-SHADE and DM-L-SHADE following each competition's rules and write the result files in the format the organizers ask for: one file `AlgorithmName_F_D.txt` per algorithm, function and dimension, with one column per run and one row per recording point. Each script also prints the per-dimension table of best, worst, median, mean and standard deviation of the final error.

| Suite | Functions | Dimensions | Runs | MaxFES | Matrix |
| --- | --- | --- | --- | --- | --- |
| CEC-2014 | 1-30 | 10, 30, 50, 100 | 51 | 10000·D | 14 × 51 |
| CEC-2017 | 1, 3-30 | 10, 30, 50, 100 | 51 | 10000·D, stops at error < 1e-8 | 14 × 51 |
| CEC-2022 | 1-12 | 10, 20 | 30 | 200,000 / 1,000,000, stops at error < 1e-8 | 17 × 30 (16 errors + FE_term) |
| CEC-2026 | 1-29 (the CEC-2017 functions) | 30 in the competition; 10, 50 and 100 also run | 25 | 10000·D | 1000 × 25 (every 10·D evaluations) |

The CEC-2022 script takes each run's seed from `data/cec22/Rand_Seeds.txt`, as the rules require; the others use `--seed` (default: current time).

Common options: `--algorithms`, `--dims`, `--functions`, `--runs`, `--seed`, `--jobs`, `--output`, `--solver`, `--force` and `--zip` (also zips each algorithm's files, ready for submission).

```
python3 experiments/cec2014_collect.py --dims 10 30
python3 experiments/cec2017_collect.py --dims 10 30 50 100 --jobs 8
python3 experiments/cec2022_collect.py --dims 10 --functions 1 2 3 --zip
python3 experiments/cec2026_collect.py --dims 10 30 --seed 2026
```

### Comparing: `compare_cec20XX_results.py`

Read the files written by the matching collect script and print, per function, the mean and standard deviation of the final error of both algorithms and a Wilcoxon rank-sum test between them; the last column says whether DM-L-SHADE is significantly better (`+`), worse (`-`) or not different (`=`) from L-SHADE, with the counts below each table. Options: `--results`, `--dims`, `--alpha`.

- **CEC-2014**: final error only (the rules define no score).
- **CEC-2017**: also computes the suite's performance measure, Score1 + Score2 (section 2.3 of the rules), weighting D = 10, 30, 50, 100 by 0.1, 0.2, 0.3, 0.4.
- **CEC-2022**: follows the suite's criterion, which rewards speed as well as accuracy: runs are ordered by final error and, among those that reached 1e-8, by the evaluations they used (FE_term). Prints each algorithm's score (U statistic) per function and in total.
- **CEC-2026**: the U-score of the competition, accuracy plus speed points over every meeting of two trials, with errors rounded to 6 significant digits (the precision of the coarsest entries' files).

```
python3 experiments/compare_cec2017_results.py
python3 experiments/compare_cec2022_results.py --dims 10 --alpha 0.01
python3 experiments/compare_cec2026_results.py --dims 10
```

### Report: `report.py`

Builds a self-contained HTML page from the same result files, using the template [templates/report.html](templates/report.html): the results tables of each dimension, the Wilcoxon comparison, the suite's score (CEC-2017, CEC-2022 and CEC-2026) and convergence curves (median and interquartile range of the error at the recording points) for every function. Dimensions are picked up from whatever files exist, so the report can be rebuilt as more results come in. It is written to `results/<suite>/report.html` by default.

`--participants` adds the ranking against the entries of the original competition:

- **CEC-2022**: the organizers' `Results_and_Ranking/Comparison` folder (their spreadsheets); the ranking is recomputed with the rule of the organizers' `ranking.m`.
- **CEC-2017**: the organizers' `Codes-of-Top-Methods-and-results.zip`, read directly (nested zip/tar.gz, and RAR through the system libarchive); adds the section 2.3 ranking and the Wilcoxon wins/ties/losses of DM-L-SHADE against each entry. This data reproduces the official Score 1 exactly, but Score 2 can differ by up to ~5 points.
- **CEC-2026**: the organizers' `Results-Data/SOP_CEC2026/SOP` folder; adds the U-score ranking at D = 30, the only dimension the entries ran, with the Wilcoxon wins/ties/losses of DM-L-SHADE against each entry. The folder has 6 of the 11 ranked entries, plus RDE, which the organizers' slides do not rank; `rde26repro_sop` is skipped because its files are identical to `rde26-sop`.

Per-suite details (function names and groups, recording points, notes) live in the `SUITES` dict at the top of the script.

```
python3 experiments/report.py --benchmark cec2014
python3 experiments/report.py --benchmark cec2017 --participants <Codes-of-Top-Methods-and-results.zip>
python3 experiments/report.py --benchmark cec2022 --participants <CEC2022 download>/Results_and_Ranking/Comparison
python3 experiments/report.py --benchmark cec2026 --participants <CEC2026 download>/Results-Data/SOP_CEC2026/SOP
```

## CVRPLIB

### Collecting: `cvrplib_collect.py`

Runs L-SHADE and DM-L-SHADE (`./solver --benchmark cvrplib`) on instances of the X set (`data/cvrplib/X`, Uchoa et al., 2017), with the same seeds for every algorithm. With `--hgs <executable>` it also runs [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) (Vidal, 2022) on the same instances and seeds, as the state-of-the-art reference. Two protocols, each in its own folder under `results/cvrplib/`:

- **evaluations** (default): 10000·n evaluations (n customers), best cost recorded at 0.01, 0.02, 0.03, 0.05, 0.1, ..., 1.0 × MaxFES. HGS-CVRP stops by its own criterion, unless `--hgs-time` gives it a CPU-time limit (seconds, or `match` for the mean time DM-L-SHADE took).
- **CPU time** (`--cpu-time tmax` or `--cpu-time <seconds>`): the protocol of Vidal (2022), every algorithm gets the same CPU time, Tmax = n × 240/100 s, with the best cost recorded at 1, 2, 5, 10, 15, 20, 30, 50, 75 and 100% of it. Use `--jobs` at most the number of physical cores.

Each line of `results/cvrplib/<protocol>/<Algorithm>/<instance>.txt` holds one run: `seed cost cpu_seconds` followed by the checkpoint costs; `protocol.json` records the settings of the folder. The summary printed at the end gives, per instance, the best and mean gap to the best known solution (the `.sol` next to the `.vrp`) and the mean CPU time, with Wilcoxon rank-sum tests; across instances, Wilcoxon signed-rank tests paired by instance (Holm-adjusted) and a Friedman test. `--summary-only` prints it from the runs on disk without running anything.

```
python3 experiments/cvrplib_collect.py --instances X-n101-k25 --runs 10 --hgs <hgs>
python3 experiments/cvrplib_collect.py --instances all --max-customers 200 --cpu-time tmax --hgs <hgs> --jobs 8
```

### Report: `cvrplib_report.py`

Builds the HTML comparison of DM-L-SHADE, L-SHADE and HGS-CVRP from the protocol folders, using [templates/cvrplib_report.html](templates/cvrplib_report.html): gap to the best known value per instance, distribution of the final gaps, convergence curves, pairwise Wilcoxon rank-sum tests and, with two or more instances, signed-rank tests (Holm-adjusted) and a Friedman test. Written to `results/cvrplib/report.html` by default; `--protocols` chooses the folders and `--hgs-dir` points to the HGS-CVRP checkout to report its version.

```
python3 experiments/cvrplib_report.py
python3 experiments/cvrplib_report.py --protocols evaluations tmax --output /tmp/cvrp.html
```

## Other scripts

- [compare_algorithms.py](compare_algorithms.py): quick comparison on a single function, outside the competition protocols. Runs both algorithms N times (`--benchmark cec2014|cec2022`, `--function`, `--runs`, `--seed`) and reports the mean and standard deviation of the final error and of the run time.

  ```
  python3 experiments/compare_algorithms.py --function 6 --runs 30
  ```

- [nonparametric.py](nonparametric.py): module, not run directly. Holds the statistical tests used by the CVRPLIB scripts: Wilcoxon rank-sum, Wilcoxon signed-rank (exact for up to 25 pairs), Friedman, and the Holm-Bonferroni adjustment (Derrac et al., 2011).
