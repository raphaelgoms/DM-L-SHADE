#ifndef _SET_COVERING_DEMO_H_
#define _SET_COVERING_DEMO_H_

#include "algorithm/algorithm_factory.h"

// Demonstrates applying the solver to a fourth discrete/combinatorial
// problem (Set Covering) via random-key encoding: the individual stays a
// real vector, decoded into a priority order over candidate subsets that is
// greedily turned into a cover only inside SetCoveringProblem's evaluate().
// The instance is built out of independent blocks with a cheap "whole"
// subset and a more expensive "two halves" alternative for each, so the
// optimal total cost (one whole subset per block) is known in closed form.
void runSetCoveringDemo(AlgorithmType algorithm_type, int n_blocks, int block_size, bool show_time);

#endif
