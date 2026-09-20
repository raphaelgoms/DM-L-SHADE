#include "demos/tsp_demo.h"

#include "problems/tsp_problem.h"

void runTSPDemo(AlgorithmType algorithm_type, int n_cities, bool show_time) {
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
