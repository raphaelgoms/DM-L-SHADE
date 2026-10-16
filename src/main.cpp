/*
  DM-L-SHADE: hybridization of L-SHADE (Success-History based Adaptive
  Differential Evolution with Linear Population Size Reduction) with Data
  Mining methods, built on top of a problem-agnostic solver so it can be
  applied to any continuous optimization problem, not only the CEC-2014
  benchmark suite.

  Version: 2.0  Date: 05/Set/2026
  Written by Raphael Gomes Santos (raphaelgoms [at] gmail.com)
*/

#include <string>

#include "algorithm/algorithm_factory.h"
#include "problems/cec14_problem.h"
#include "problems/cec17_problem.h"
#include "problems/cec22_problem.h"
#include "cvrplib/cvrplib_solver.h"
#include "demos/sphere_demo.h"
#include "demos/tsp_demo.h"
#include "demos/knapsack_demo.h"
#include "demos/cvrp_demo.h"
#include "demos/set_covering_demo.h"

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

// Benchmark suite that --f indexes into.
enum class BenchmarkType { CEC2014, CEC2017, CEC2022 };

// Number of functions defined by each benchmark suite; bounds what --f and
// the default (whole-suite) function range accept.
static int benchmarkFunctionCount(BenchmarkType benchmark_type) {
  return benchmark_type == BenchmarkType::CEC2022 ? 12 : 30;
}

// Whether a function number within 1..benchmarkFunctionCount() is part of the
// suite: CEC-2017 removed F2, but keeps the original numbering of the others.
static bool benchmarkHasFunction(BenchmarkType benchmark_type, int function_number) {
  return !(benchmark_type == BenchmarkType::CEC2017 && function_number == 2);
}

// Dimensions each benchmark suite defines (and ships data files for).
static bool benchmarkSupportsDimension(BenchmarkType benchmark_type, int dimension) {
  if (benchmark_type == BenchmarkType::CEC2022) return dimension == 10 || dimension == 20;
  return dimension == 10 || dimension == 30 || dimension == 50 || dimension == 100;
}

// CEC-2014 and CEC-2017 rules: record the error at these fractions of MaxFES in every run.
static const double kCheckpointFractions[] = {
  0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
};

// MaxFES of each suite: 10000*D for CEC-2014 and CEC-2017; fixed per
// dimension for CEC-2022.
static unsigned int benchmarkMaxEvaluations(BenchmarkType benchmark_type, int dimension) {
  if (benchmark_type == BenchmarkType::CEC2022) return dimension == 20 ? 1000000 : 200000;
  return dimension * 10000;
}

// Evaluation counts at which each suite's rules record the error: the
// fractions above of MaxFES for CEC-2014 and CEC-2017, and floor(D^(k/5-3) * MaxFES),
// k = 0..15, for CEC-2022.
static vector<unsigned int> benchmarkCheckpoints(BenchmarkType benchmark_type, int dimension, unsigned int max_evaluations) {
  vector<unsigned int> checkpoints;
  if (benchmark_type == BenchmarkType::CEC2022) {
    for (int k = 0; k <= 15; k++) {
      // The small offset keeps exact products (e.g. 10^-2 * 200000) from
      // flooring one below because of rounding in pow().
      checkpoints.push_back((unsigned int)floor(pow(dimension, k / 5.0 - 3) * max_evaluations + 1e-6));
    }
  } else {
    for (double fraction : kCheckpointFractions) {
      checkpoints.push_back((unsigned int)llround(fraction * max_evaluations));
    }
  }
  return checkpoints;
}

static shared_ptr<Problem> createBenchmarkProblem(BenchmarkType benchmark_type, int function_number, int problem_size) {
  switch (benchmark_type) {
    case BenchmarkType::CEC2022:
      return make_shared<CEC22Problem>(function_number, problem_size);
    case BenchmarkType::CEC2017:
      return make_shared<CEC17Problem>(function_number, problem_size);
    case BenchmarkType::CEC2014:
    default:
      return make_shared<CEC14Problem>(function_number, problem_size);
  }
}

static void runBenchmark(BenchmarkType benchmark_type, const AlgorithmSpec &algorithm, int function_start, int function_end, int problem_size, int num_runs, bool show_time, bool record_checkpoints) {
  SHADEConfig config;
  config.pop_size = (int)round(problem_size * 18);
  config.max_num_evaluations = benchmarkMaxEvaluations(benchmark_type, problem_size);
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;
  // CEC-2017 and CEC-2022 rules: a run ends when the error drops below 1e-8.
  config.stop_at_optimum = benchmark_type == BenchmarkType::CEC2017 ||
                           benchmark_type == BenchmarkType::CEC2022;
  if (record_checkpoints) {
    config.checkpoint_evaluations = benchmarkCheckpoints(benchmark_type, problem_size, config.max_num_evaluations);
  }

  // A single run prints only its error value, so callers (e.g. the scripts
  // in experiments/) can parse the output without stripping the report.
  bool single_run = num_runs == 1;

  for (int function_number = function_start; function_number <= function_end; function_number++) {
    if (!benchmarkHasFunction(benchmark_type, function_number)) continue;
    if (!single_run) {
      cout << "\n-------------------------------------------------------" << endl;
      cout << "Function = " << function_number << ", Dimension size = " << problem_size << "\n" << endl;
    }

    Fitness *bsf_fitness_array = (Fitness*)malloc(sizeof(Fitness) * num_runs);
    Fitness mean_bsf_fitness = 0;
    Fitness std_bsf_fitness = 0;

    for (int j = 0; j < num_runs; j++) {
      auto problem = createBenchmarkProblem(benchmark_type, function_number, problem_size);

      searchAlgorithm *alg = createAlgorithm(algorithm, problem, config);
      bsf_fitness_array[j] = runAlgorithm(alg, show_time);
      if (record_checkpoints) {
        // One line with the error at each checkpoint; the last one is the
        // final error. CEC-2022 also records the evaluations used (FE_term).
        const vector<Fitness> &checkpoint_errors = alg->checkpointErrors();
        for (size_t k = 0; k < checkpoint_errors.size(); k++) {
          cout << (k ? " " : "") << checkpoint_errors[k];
        }
        if (benchmark_type == BenchmarkType::CEC2022) cout << " " << alg->terminationEvaluations();
        cout << endl;
      }
      else if (single_run) cout << bsf_fitness_array[j] << endl;
      else cout << j + 1 << "th run, " << "error value = " << bsf_fitness_array[j] << endl;
      delete alg;
    }

    if (!single_run) {
      for (int j = 0; j < num_runs; j++) mean_bsf_fitness += bsf_fitness_array[j];
      mean_bsf_fitness /= num_runs;

      for (int j = 0; j < num_runs; j++) std_bsf_fitness += pow((mean_bsf_fitness - bsf_fitness_array[j]), 2.0);
      std_bsf_fitness /= num_runs;
      std_bsf_fitness = sqrt(std_bsf_fitness);

      cout  << "\nmean = " << mean_bsf_fitness << ", std = " << std_bsf_fitness << endl;
    }
    free(bsf_fitness_array);
  }
}

int main(int argc, char **argv) {
  bool sphere_demo = false;
  bool tsp_demo = false;
  int tsp_n_cities = 15;
  bool knapsack_demo = false;
  int knapsack_n_items = 20;
  bool cvrp_demo = false;
  int cvrp_n_clusters = 5;
  int cvrp_cluster_size = 4;
  bool set_covering_demo = false;
  int set_covering_n_blocks = 8;
  int set_covering_block_size = 4;
  bool show_time = false;
  int num_runs = 51;
  int problem_size = 10;
  bool record_checkpoints = false;
  bool seed_explicit = false;
  unsigned seed = 0;
  int function_start = 1;
  int function_end = 30;
  bool function_range_explicit = false;
  int requested_function = 1;
  AlgorithmSpec algorithm;  // L-SHADE unless --algorithm says otherwise
  BenchmarkType benchmark_type = BenchmarkType::CEC2014;
  // --benchmark cvrplib solves a CVRPLIB instance instead of a CEC suite.
  bool cvrplib = false;
  string instance_path;
  string solution_path;
  double cpu_seconds = 0;  // --cpu-time; negative means "tmax"
  CVRPLibSolverOptions cvrplib_options;  // --pop-rate, --memory-size, --arc-rate, --p-best-rate
  bool shade_parameters_explicit = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sphere") == 0) {
      sphere_demo = true;
    } else if (strcmp(argv[i], "--tsp") == 0) {
      tsp_demo = true;
    } else if (strcmp(argv[i], "--tsp-cities") == 0 && i + 1 < argc) {
      tsp_n_cities = atoi(argv[i + 1]);
      if (tsp_n_cities < 3) {
        cerr << "Invalid number of cities. Please use a value >= 3." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--knapsack") == 0) {
      knapsack_demo = true;
    } else if (strcmp(argv[i], "--knapsack-items") == 0 && i + 1 < argc) {
      knapsack_n_items = atoi(argv[i + 1]);
      if (knapsack_n_items < 1) {
        cerr << "Invalid number of items. Please use a value >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--cvrp") == 0) {
      cvrp_demo = true;
    } else if (strcmp(argv[i], "--cvrp-clusters") == 0 && i + 1 < argc) {
      cvrp_n_clusters = atoi(argv[i + 1]);
      if (cvrp_n_clusters < 1) {
        cerr << "Invalid number of clusters. Please use a value >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--cvrp-cluster-size") == 0 && i + 1 < argc) {
      cvrp_cluster_size = atoi(argv[i + 1]);
      if (cvrp_cluster_size < 1) {
        cerr << "Invalid cluster size. Please use a value >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--setcover") == 0) {
      set_covering_demo = true;
    } else if (strcmp(argv[i], "--setcover-blocks") == 0 && i + 1 < argc) {
      set_covering_n_blocks = atoi(argv[i + 1]);
      if (set_covering_n_blocks < 1) {
        cerr << "Invalid number of blocks. Please use a value >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--setcover-block-size") == 0 && i + 1 < argc) {
      set_covering_block_size = atoi(argv[i + 1]);
      if (set_covering_block_size < 2) {
        cerr << "Invalid block size. Please use a value >= 2." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--show-time") == 0) {
      show_time = true;
    } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      num_runs = atoi(argv[i + 1]);
      if (num_runs < 1) {
        cerr << "Invalid number of runs. Please use a value >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
      // Which sizes are valid depends on the benchmark suite, checked after
      // the argument loop.
      problem_size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "--checkpoints") == 0) {
      record_checkpoints = true;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = (unsigned)strtoul(argv[i + 1], NULL, 10);
      seed_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
      string benchmark_name = argv[i + 1];
      for (auto &c : benchmark_name) c = tolower(c);

      if (benchmark_name == "cec2014" || benchmark_name == "cec14") {
        benchmark_type = BenchmarkType::CEC2014;
      } else if (benchmark_name == "cec2017" || benchmark_name == "cec17") {
        benchmark_type = BenchmarkType::CEC2017;
      } else if (benchmark_name == "cec2022" || benchmark_name == "cec22") {
        benchmark_type = BenchmarkType::CEC2022;
      } else if (benchmark_name == "cvrplib") {
        cvrplib = true;
      } else {
        cerr << "Invalid benchmark. Please use \"cec2014\", \"cec2017\", \"cec2022\" or \"cvrplib\"." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--elite-rate") == 0 && i + 1 < argc) {
      algorithm.dmlshade.elite_rate = atof(argv[i + 1]);
      if (algorithm.dmlshade.elite_rate <= 0 || algorithm.dmlshade.elite_rate > 1) {
        cerr << "Invalid elite rate. Please use a value in (0, 1]." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--clusters-rate") == 0 && i + 1 < argc) {
      // <= 0 lets X-means choose the number of clusters.
      algorithm.dmlshade.clusters_rate = atof(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "--mining-step") == 0 && i + 1 < argc) {
      algorithm.dmlshade.mining_generation_step = atoi(argv[i + 1]);
      if (algorithm.dmlshade.mining_generation_step < 1) {
        cerr << "Invalid mining step. Please use a number of generations >= 1." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--ls-rate") == 0 && i + 1 < argc) {
      algorithm.dmlshade.local_search_rate = atof(argv[i + 1]);
      // 0 turns the local search off: the patterns are only evaluated.
      if (algorithm.dmlshade.local_search_rate < 0) {
        cerr << "Invalid local search rate. Please use a value >= 0." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--ls-method") == 0 && i + 1 < argc) {
      string method_name = argv[i + 1];
      for (auto &c : method_name) c = tolower(c);

      if (method_name == "es") {
        algorithm.dmlshade.local_search_method = LocalSearchMethod::ES;
      } else if (method_name == "hooke-jeeves" || method_name == "hj") {
        algorithm.dmlshade.local_search_method = LocalSearchMethod::HOOKE_JEEVES;
      } else {
        cerr << "Invalid local search method. Please use \"es\" or \"hooke-jeeves\"." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
      instance_path = argv[i + 1];
      i++;
    } else if (strcmp(argv[i], "--cpu-time") == 0 && i + 1 < argc) {
      if (strcmp(argv[i + 1], "tmax") == 0) {
        cpu_seconds = -1;
      } else {
        cpu_seconds = atof(argv[i + 1]);
        if (cpu_seconds <= 0) {
          cerr << "Invalid CPU time. Please use a number of seconds > 0 or \"tmax\"." << endl;
          return 1;
        }
      }
      i++;
    } else if (strcmp(argv[i], "--pop-rate") == 0 && i + 1 < argc) {
      cvrplib_options.pop_rate = atof(argv[i + 1]);
      if (cvrplib_options.pop_rate <= 0) {
        cerr << "Invalid population rate. Please use a value > 0." << endl;
        return 1;
      }
      shade_parameters_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--memory-size") == 0 && i + 1 < argc) {
      cvrplib_options.memory_size = atoi(argv[i + 1]);
      if (cvrplib_options.memory_size < 1) {
        cerr << "Invalid memory size. Please use a value >= 1." << endl;
        return 1;
      }
      shade_parameters_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--arc-rate") == 0 && i + 1 < argc) {
      cvrplib_options.arc_rate = atof(argv[i + 1]);
      if (cvrplib_options.arc_rate < 0) {
        cerr << "Invalid archive rate. Please use a value >= 0." << endl;
        return 1;
      }
      shade_parameters_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--p-best-rate") == 0 && i + 1 < argc) {
      cvrplib_options.p_best_rate = atof(argv[i + 1]);
      if (cvrplib_options.p_best_rate <= 0 || cvrplib_options.p_best_rate > 1) {
        cerr << "Invalid p-best rate. Please use a value in (0, 1]." << endl;
        return 1;
      }
      shade_parameters_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--solution") == 0 && i + 1 < argc) {
      solution_path = argv[i + 1];
      i++;
    } else if (strcmp(argv[i], "--f") == 0 && i + 1 < argc) {
      // Range depends on the benchmark suite, which may be parsed after
      // this flag, so the check is deferred until after the argument loop.
      requested_function = atoi(argv[i + 1]);
      function_range_explicit = true;
      i++;
    } else if (strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
      string algorithm_name = argv[i + 1];
      for (auto &c : algorithm_name) c = tolower(c);

      if (algorithm_name == "lshade") {
        algorithm.type = AlgorithmType::LSHADE;
      } else if (algorithm_name == "dmlshade" || algorithm_name == "dm-lshade") {
        algorithm.type = AlgorithmType::DMLSHADE;
      } else if (algorithm_name == "dmlshade-ls" || algorithm_name == "dm-lshade-ls") {
        algorithm.type = AlgorithmType::DMLSHADE_LS;
      } else {
        cerr << "Invalid algorithm. Please use \"lshade\", \"dmlshade\" or \"dmlshade-ls\"." << endl;
        return 1;
      }
      i++;
    }
  }

  // Random seed is selected based on time according to competition rules,
  // unless --seed is given to make a run reproducible (or to give concurrent
  // runs started within the same second different seeds).
  srand(seed_explicit ? seed : (unsigned)time(NULL));

  int max_function_number = benchmarkFunctionCount(benchmark_type);
  if (function_range_explicit) {
    if (requested_function < 1 || requested_function > max_function_number) {
      cerr << "Invalid function number. Please use a value between 1 and "
           << max_function_number << "." << endl;
      return 1;
    }
    if (!benchmarkHasFunction(benchmark_type, requested_function)) {
      cerr << "Function " << requested_function << " is not part of this benchmark"
           << " (CEC-2017 removed F2)." << endl;
      return 1;
    }
    function_start = function_end = requested_function;
  } else {
    function_start = 1;
    function_end = max_function_number;
  }

  if (record_checkpoints && num_runs != 1) {
    cerr << "--checkpoints requires --runs 1." << endl;
    return 1;
  }

  if (shade_parameters_explicit && !cvrplib) {
    cerr << "--pop-rate, --memory-size, --arc-rate and --p-best-rate apply to --benchmark cvrplib only." << endl;
    return 1;
  }

  if (cvrplib) {
    if (instance_path.empty()) {
      cerr << "--benchmark cvrplib requires --instance <file.vrp>." << endl;
      return 1;
    }
    CVRPLibSolverOptions options = cvrplib_options;
    options.instance_path = instance_path;
    options.solution_path = solution_path;
    options.num_runs = num_runs;
    options.show_time = show_time;
    options.record_checkpoints = record_checkpoints;
    options.cpu_seconds = cpu_seconds;
    try {
      runCVRPLibSolver(algorithm, options);
    } catch (const std::exception &e) {
      cerr << "Error: " << e.what() << endl;
      return 1;
    }
    return 0;
  }

  if (sphere_demo) {
    runSphereDemo(algorithm, problem_size, show_time);
    return 0;
  }

  if (tsp_demo) {
    runTSPDemo(algorithm, tsp_n_cities, show_time);
    return 0;
  }

  if (knapsack_demo) {
    runKnapsackDemo(algorithm, knapsack_n_items, show_time);
    return 0;
  }

  if (cvrp_demo) {
    runCVRPDemo(algorithm, cvrp_n_clusters, cvrp_cluster_size, show_time);
    return 0;
  }

  if (set_covering_demo) {
    runSetCoveringDemo(algorithm, set_covering_n_blocks, set_covering_block_size, show_time);
    return 0;
  }

  if (!benchmarkSupportsDimension(benchmark_type, problem_size)) {
    cerr << "Invalid dimension for this benchmark. Please use "
         << (benchmark_type == BenchmarkType::CEC2022 ? "10 or 20" : "10, 30, 50 or 100") << "." << endl;
    return 1;
  }

  runBenchmark(benchmark_type, algorithm, function_start, function_end, problem_size, num_runs, show_time, record_checkpoints);

  return 0;
}
