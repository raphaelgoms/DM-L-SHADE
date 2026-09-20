#include "demos/set_covering_demo.h"

#include "problems/set_covering_problem.h"

void runSetCoveringDemo(AlgorithmType algorithm_type, int n_blocks, int block_size, bool show_time) {
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
