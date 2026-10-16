#ifndef _ALGORITHM_FACTORY_H_
#define _ALGORITHM_FACTORY_H_

#include "algorithm/algorithm.h"

enum class AlgorithmType { LSHADE, DMLSHADE, DMLSHADE_LS };

// Parameters of DM-L-SHADE's pattern mining (unused by L-SHADE). The
// defaults are the tuned values used in all the reported experiments; the
// CLI can override them (--elite-rate, --clusters-rate, --mining-step).
struct DMLSHADEParameters {
  // Elite set size as a fraction of the initial population.
  double elite_rate = 0.1;
  // Number of clusters (patterns) as a fraction of the elite size; <= 0
  // lets X-means choose the number of clusters instead of K-means.
  double clusters_rate = 0.1468;
  // Mine the elite every this many generations.
  int mining_generation_step = 168;
  // DM-L-SHADE-LS only: evaluations of the local search around each pattern,
  // as a multiple of the dimension (--ls-rate); 0 only evaluates the patterns.
  double local_search_rate = 10;
  // DM-L-SHADE-LS only: which local search (--ls-method es|hooke-jeeves).
  LocalSearchMethod local_search_method = LocalSearchMethod::ES;
};

// Which algorithm to build, with its algorithm-specific parameters.
struct AlgorithmSpec {
  AlgorithmType type = AlgorithmType::LSHADE;
  DMLSHADEParameters dmlshade;
};

searchAlgorithm *createAlgorithm(const AlgorithmSpec &algorithm, shared_ptr<Problem> problem, const SHADEConfig &config);

// Runs the algorithm and, if requested, prints how long it took. Kept off by
// default since timing isn't meaningful when comparing across machines/runs.
Fitness runAlgorithm(searchAlgorithm *alg, bool show_time);

#endif
