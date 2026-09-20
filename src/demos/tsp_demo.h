#ifndef _TSP_DEMO_H_
#define _TSP_DEMO_H_

#include "algorithm/algorithm_factory.h"

// Demonstrates applying the solver to a discrete/combinatorial problem
// (Euclidean TSP) via random-key encoding: the individual stays a real
// vector, decoded into a permutation of cities only inside TSPProblem's
// evaluate(). Cities are placed on a circle so the optimal tour length is
// known in closed form, letting us check how close the solver gets to it.
void runTSPDemo(AlgorithmType algorithm_type, int n_cities, bool show_time);

#endif
