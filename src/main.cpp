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
#include "problems/cec22_problem.h"
#include "demos/sphere_demo.h"
#include "demos/tsp_demo.h"
#include "demos/knapsack_demo.h"
#include "demos/cvrp_demo.h"
#include "demos/set_covering_demo.h"

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

// Benchmark suite that --f indexes into.
enum class BenchmarkType { CEC2014, CEC2022 };

// Number of functions defined by each benchmark suite; bounds what --f and
// the default (whole-suite) function range accept.
static int benchmarkFunctionCount(BenchmarkType benchmark_type) {
  return benchmark_type == BenchmarkType::CEC2022 ? 12 : 30;
}

static shared_ptr<Problem> createBenchmarkProblem(BenchmarkType benchmark_type, int function_number, int problem_size) {
  switch (benchmark_type) {
    case BenchmarkType::CEC2022:
      return make_shared<CEC22Problem>(function_number, problem_size);
    case BenchmarkType::CEC2014:
    default:
      return make_shared<CEC14Problem>(function_number, problem_size);
  }
}

static void runBenchmark(BenchmarkType benchmark_type, AlgorithmType algorithm_type, int function_start, int function_end, int problem_size, int num_runs, bool show_time) {
  SHADEConfig config;
  config.pop_size = (int)round(problem_size * 18);
  config.max_num_evaluations = problem_size * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  for (int function_number = function_start; function_number <= function_end; function_number++) {
    cout << "\n-------------------------------------------------------" << endl;
    cout << "Function = " << function_number << ", Dimension size = " << problem_size << "\n" << endl;

    Fitness *bsf_fitness_array = (Fitness*)malloc(sizeof(Fitness) * num_runs);
    Fitness mean_bsf_fitness = 0;
    Fitness std_bsf_fitness = 0;

    for (int j = 0; j < num_runs; j++) {
      auto problem = createBenchmarkProblem(benchmark_type, function_number, problem_size);

      searchAlgorithm *alg = createAlgorithm(algorithm_type, problem, config);
      bsf_fitness_array[j] = runAlgorithm(alg, show_time);
      cout << j + 1 << "th run, " << "error value = " << bsf_fitness_array[j] << endl;
      delete alg;
    }

    for (int j = 0; j < num_runs; j++) mean_bsf_fitness += bsf_fitness_array[j];
    mean_bsf_fitness /= num_runs;

    for (int j = 0; j < num_runs; j++) std_bsf_fitness += pow((mean_bsf_fitness - bsf_fitness_array[j]), 2.0);
    std_bsf_fitness /= num_runs;
    std_bsf_fitness = sqrt(std_bsf_fitness);

    cout  << "\nmean = " << mean_bsf_fitness << ", std = " << std_bsf_fitness << endl;
    free(bsf_fitness_array);
  }
}

int main(int argc, char **argv) {
  //random seed is selected based on time according to competition rules
  srand((unsigned)time(NULL));

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
  int function_start = 1;
  int function_end = 30;
  bool function_range_explicit = false;
  int requested_function = 1;
  AlgorithmType algorithm_type = AlgorithmType::LSHADE;
  BenchmarkType benchmark_type = BenchmarkType::CEC2014;

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
    } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
      string benchmark_name = argv[i + 1];
      for (auto &c : benchmark_name) c = tolower(c);

      if (benchmark_name == "cec2014" || benchmark_name == "cec14") {
        benchmark_type = BenchmarkType::CEC2014;
      } else if (benchmark_name == "cec2022" || benchmark_name == "cec22") {
        benchmark_type = BenchmarkType::CEC2022;
      } else {
        cerr << "Invalid benchmark. Please use \"cec2014\" or \"cec2022\"." << endl;
        return 1;
      }
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
        algorithm_type = AlgorithmType::LSHADE;
      } else if (algorithm_name == "dmlshade" || algorithm_name == "dm-lshade") {
        algorithm_type = AlgorithmType::DMLSHADE;
      } else {
        cerr << "Invalid algorithm. Please use \"lshade\" or \"dmlshade\"." << endl;
        return 1;
      }
      i++;
    }
  }

  int max_function_number = benchmarkFunctionCount(benchmark_type);
  if (function_range_explicit) {
    if (requested_function < 1 || requested_function > max_function_number) {
      cerr << "Invalid function number. Please use a value between 1 and "
           << max_function_number << "." << endl;
      return 1;
    }
    function_start = function_end = requested_function;
  } else {
    function_start = 1;
    function_end = max_function_number;
  }

  //dimension size. please select from 10, 30, 50, 100
  int problem_size = 10;

  if (sphere_demo) {
    runSphereDemo(algorithm_type, problem_size, show_time);
    return 0;
  }

  if (tsp_demo) {
    runTSPDemo(algorithm_type, tsp_n_cities, show_time);
    return 0;
  }

  if (knapsack_demo) {
    runKnapsackDemo(algorithm_type, knapsack_n_items, show_time);
    return 0;
  }

  if (cvrp_demo) {
    runCVRPDemo(algorithm_type, cvrp_n_clusters, cvrp_cluster_size, show_time);
    return 0;
  }

  if (set_covering_demo) {
    runSetCoveringDemo(algorithm_type, set_covering_n_blocks, set_covering_block_size, show_time);
    return 0;
  }

  //number of runs
  int num_runs = 51;
  runBenchmark(benchmark_type, algorithm_type, function_start, function_end, problem_size, num_runs, show_time);

  return 0;
}
