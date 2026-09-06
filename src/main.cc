/*
  DM-L-SHADE: hybridization of L-SHADE (Success-History based Adaptive
  Differential Evolution with Linear Population Size Reduction) with Data
  Mining methods, built on top of a problem-agnostic solver so it can be
  applied to any continuous optimization problem, not only the CEC-2014
  benchmark suite.

  Version: 2.0  Date: 05/Set/2026
  Written by Raphael Gomes Santos (raphaelgoms [at] gmail.com)
*/

#include "de.h"
#include "cec14_problem.h"

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

static void runCEC14Benchmark(int function_start, int function_end, int problem_size, int num_runs) {
  //DM-L-SHADE parameters
  double elite_rate = 0.1;
  double clusters_rate = 0.1468;
  int mining_generation_step = 168;

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
      auto problem = make_shared<CEC14Problem>(function_number, problem_size);

      int max_elite_size = std::round(elite_rate * config.pop_size);
      int number_of_patterns = std::round(clusters_rate * max_elite_size);
      searchAlgorithm *alg = new DMLSHADE(problem, config, max_elite_size, number_of_patterns, mining_generation_step);
      bsf_fitness_array[j] = alg->run();
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
static void runSphereDemo(int problem_size) {
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

  searchAlgorithm *alg = new LSHADE(sphere, config);
  Fitness result = alg->run();
  cout << "best fitness found = " << result << endl;
  delete alg;
}

int main(int argc, char **argv) {
  //random seed is selected based on time according to competition rules
  srand((unsigned)time(NULL));

  bool sphere_demo = false;
  int function_start = 1;
  int function_end = 30;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sphere") == 0) {
      sphere_demo = true;
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
    }
  }

  //dimension size. please select from 10, 30, 50, 100
  int problem_size = 10;

  if (sphere_demo) {
    runSphereDemo(problem_size);
    return 0;
  }

  //number of runs
  int num_runs = 51;
  runCEC14Benchmark(function_start, function_end, problem_size, num_runs);

  return 0;
}
