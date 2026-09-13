/*
  DM-L-SHADE: hybridization of L-SHADE (Success-History based Adaptive
  Differential Evolution with Linear Population Size Reduction) with Data
  Mining methods, built on top of a problem-agnostic solver so it can be
  applied to any continuous optimization problem, not only the CEC-2014
  benchmark suite.

  Version: 2.0  Date: 05/Set/2026
  Written by Raphael Gomes Santos (raphaelgoms [at] gmail.com)
*/

#include <chrono>
#include <string>

#include "de.h"
#include "problems/cec14_problem.h"
#include "problems/tsp_problem.h"
#include "problems/knapsack_problem.h"
#include "problems/cvrp_problem.h"
#include "problems/set_covering_problem.h"

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

enum class AlgorithmType { LSHADE, DMLSHADE };

// Benchmark suite that --f indexes into. Only CEC2014 exists today, but this
// is where future suites (e.g. CEC2017, CEC2020) will be added.
enum class BenchmarkType { CEC2014 };

static searchAlgorithm *createAlgorithm(AlgorithmType algorithm_type, shared_ptr<Problem> problem, const SHADEConfig &config) {
  if (algorithm_type == AlgorithmType::DMLSHADE) {
    //DM-L-SHADE parameters
    double elite_rate = 0.1;
    double clusters_rate = 0.1468;
    int mining_generation_step = 168;

    int max_elite_size = std::round(elite_rate * config.pop_size);
    int number_of_patterns = std::round(clusters_rate * max_elite_size);
    return new DMLSHADE(problem, config, max_elite_size, number_of_patterns, mining_generation_step);
  }

  return new LSHADE(problem, config);
}

// Runs the algorithm and, if requested, prints how long it took. Kept off by
// default since timing isn't meaningful when comparing across machines/runs.
static Fitness runAlgorithm(searchAlgorithm *alg, bool show_time) {
  if (!show_time) return alg->run();

  auto start = std::chrono::steady_clock::now();
  Fitness result = alg->run();
  auto end = std::chrono::steady_clock::now();

  double elapsed_seconds = std::chrono::duration<double>(end - start).count();
  cout << "execution time = " << elapsed_seconds << " s" << endl;

  return result;
}

static shared_ptr<Problem> createBenchmarkProblem(BenchmarkType benchmark_type, int function_number, int problem_size) {
  switch (benchmark_type) {
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

// Demonstrates that the solver is not tied to the CEC14 benchmark: optimizes
// the Sphere function (sum of x_i^2) via a FunctionProblem, never touching
// cec14_test_func.
static void runSphereDemo(AlgorithmType algorithm_type, int problem_size, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "Sphere function demo, Dimension size = " << problem_size << "\n" << endl;

  auto sphere = make_shared<FunctionProblem>(problem_size, -100.0, 100.0,
    [](const double *x, int n) {
      double sum = 0;
      for (int i = 0; i < n; i++) sum += x[i] * x[i];
      return sum;
    });

  SHADEConfig config;
  config.pop_size = (int)round(problem_size * 18);
  config.max_num_evaluations = problem_size * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, sphere, config);
  Fitness result = runAlgorithm(alg, show_time);
  cout << "best fitness found = " << result << endl;
  delete alg;
}

// Demonstrates applying the solver to a discrete/combinatorial problem
// (Euclidean TSP) via random-key encoding: the individual stays a real
// vector, decoded into a permutation of cities only inside TSPProblem's
// evaluate(). Cities are placed on a circle so the optimal tour length is
// known in closed form, letting us check how close the solver gets to it.
static void runTSPDemo(AlgorithmType algorithm_type, int n_cities, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "TSP demo (random-key encoding), number of cities = " << n_cities << "\n" << endl;

  auto tsp = makeCircleTSPInstance(n_cities, 100.0);

  SHADEConfig config;
  config.pop_size = (int)round(n_cities * 18);
  config.max_num_evaluations = n_cities * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, tsp, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "known optimal tour length = " << tsp->knownOptimum() << endl;
  cout << "best tour length found    = " << tsp->knownOptimum() + gap_to_optimum << endl;
  cout << "gap to known optimum      = " << gap_to_optimum << endl;
}

// Demonstrates applying the solver to another discrete/combinatorial problem
// (0/1 knapsack) via random-key encoding: the individual stays a real
// vector, decoded into a greedily packed item set only inside
// KnapsackProblem's evaluate(). The instance is built so the optimal packed
// value is known in closed form, letting us check how close the solver gets.
static void runKnapsackDemo(AlgorithmType algorithm_type, int n_items, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "Knapsack demo (random-key encoding), number of items = " << n_items << "\n" << endl;

  auto knapsack = makeSubsetSumKnapsackInstance(n_items);

  SHADEConfig config;
  config.pop_size = (int)round(n_items * 18);
  config.max_num_evaluations = n_items * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, knapsack, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  Fitness known_best_value = -knapsack->knownOptimum();
  Fitness best_value_found = known_best_value - gap_to_optimum;

  cout << "capacity                = " << knapsack->capacity() << endl;
  cout << "known optimal value     = " << known_best_value << endl;
  cout << "best value found        = " << best_value_found << endl;
  cout << "gap to known optimum    = " << gap_to_optimum << endl;
}

// Demonstrates applying the solver to a third discrete/combinatorial problem
// (Capacitated Vehicle Routing) via random-key encoding: the individual
// stays a real vector, decoded into a giant tour that is greedily cut into
// vehicle routes only inside CVRPProblem's evaluate(). The instance is built
// out of well-separated, capacity-matched customer clusters so the optimal
// total distance (one dedicated route per cluster) is known in closed form.
static void runCVRPDemo(AlgorithmType algorithm_type, int n_clusters, int customers_per_cluster, bool show_time) {
  int n_customers = n_clusters * customers_per_cluster;

  cout << "\n-------------------------------------------------------" << endl;
  cout << "CVRP demo (random-key encoding), " << n_clusters << " clusters x "
       << customers_per_cluster << " customers = " << n_customers << " customers\n" << endl;

  auto cvrp = makeClusteredCVRPInstance(n_clusters, customers_per_cluster, 100.0);

  SHADEConfig config;
  config.pop_size = (int)round(n_customers * 18);
  config.max_num_evaluations = n_customers * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, cvrp, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "known optimal total distance = " << cvrp->knownOptimum() << endl;
  cout << "best total distance found    = " << cvrp->knownOptimum() + gap_to_optimum << endl;
  cout << "gap to known optimum         = " << gap_to_optimum << endl;
}

// Demonstrates applying the solver to a fourth discrete/combinatorial
// problem (Set Covering) via random-key encoding: the individual stays a
// real vector, decoded into a priority order over candidate subsets that is
// greedily turned into a cover only inside SetCoveringProblem's evaluate().
// The instance is built out of independent blocks with a cheap "whole"
// subset and a more expensive "two halves" alternative for each, so the
// optimal total cost (one whole subset per block) is known in closed form.
static void runSetCoveringDemo(AlgorithmType algorithm_type, int n_blocks, int block_size, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "Set Covering demo (random-key encoding), " << n_blocks << " blocks x "
       << block_size << " elements\n" << endl;

  auto set_covering = makeBlockedSetCoveringInstance(n_blocks, block_size);
  int n_subsets = set_covering->dimension();

  SHADEConfig config;
  config.pop_size = (int)round(n_subsets * 18);
  config.max_num_evaluations = n_subsets * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, set_covering, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "known optimal cover cost = " << set_covering->knownOptimum() << endl;
  cout << "best cover cost found    = " << set_covering->knownOptimum() + gap_to_optimum << endl;
  cout << "gap to known optimum     = " << gap_to_optimum << endl;
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
      } else {
        cerr << "Invalid benchmark. Please use \"cec2014\"." << endl;
        return 1;
      }
      i++;
    } else if (strcmp(argv[i], "--f") == 0 && i + 1 < argc) {
      int func_num = atoi(argv[i + 1]);
      if (func_num >= 1 && func_num <= 30) {
        function_start = func_num;
        function_end = func_num;
      } else {
        cerr << "Invalid function number. Please use a value between 1 and 30." << endl;
        return 1;
      }
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
