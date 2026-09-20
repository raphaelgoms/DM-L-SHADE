#include "demos/knapsack_demo.h"

#include "problems/knapsack_problem.h"

void runKnapsackDemo(AlgorithmType algorithm_type, int n_items, bool show_time) {
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
