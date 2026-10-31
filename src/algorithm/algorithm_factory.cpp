#include "algorithm/algorithm_factory.h"

#include <chrono>

searchAlgorithm *createAlgorithm(const AlgorithmSpec &algorithm, shared_ptr<Problem> problem, const SHADEConfig &config) {
  if (algorithm.type == AlgorithmType::DMLSHADE) {
    const DMLSHADEParameters &params = algorithm.dmlshade;
    int max_elite_size = std::round(params.elite_rate * config.pop_size);
    int number_of_patterns = std::round(params.clusters_rate * max_elite_size);
    return new DMLSHADE(problem, config, max_elite_size, number_of_patterns, params.mining_generation_step,
                        params.pattern_strategy, params.pattern_pbest_generations);
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
