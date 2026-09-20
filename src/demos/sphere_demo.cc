#include "demos/sphere_demo.h"

void runSphereDemo(AlgorithmType algorithm_type, int problem_size, bool show_time) {
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
