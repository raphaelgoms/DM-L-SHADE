#include "algorithm/algorithm_factory.h"

#include <chrono>

searchAlgorithm *createAlgorithm(AlgorithmType algorithm_type, shared_ptr<Problem> problem, const SHADEConfig &config) {
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

Fitness runAlgorithm(searchAlgorithm *alg, bool show_time) {
  if (!show_time) return alg->run();

  auto start = std::chrono::steady_clock::now();
  Fitness result = alg->run();
  auto end = std::chrono::steady_clock::now();

  double elapsed_seconds = std::chrono::duration<double>(end - start).count();
  cout << "execution time = " << elapsed_seconds << " s" << endl;

  return result;
}
