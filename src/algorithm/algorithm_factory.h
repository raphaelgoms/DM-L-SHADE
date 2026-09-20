#ifndef _ALGORITHM_FACTORY_H_
#define _ALGORITHM_FACTORY_H_

#include "algorithm/algorithm.h"

enum class AlgorithmType { LSHADE, DMLSHADE };

searchAlgorithm *createAlgorithm(AlgorithmType algorithm_type, shared_ptr<Problem> problem, const SHADEConfig &config);

// Runs the algorithm and, if requested, prints how long it took. Kept off by
// default since timing isn't meaningful when comparing across machines/runs.
Fitness runAlgorithm(searchAlgorithm *alg, bool show_time);

#endif
