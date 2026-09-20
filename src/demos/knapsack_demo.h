#ifndef _KNAPSACK_DEMO_H_
#define _KNAPSACK_DEMO_H_

#include "algorithm/algorithm_factory.h"

// Demonstrates applying the solver to another discrete/combinatorial problem
// (0/1 knapsack) via random-key encoding: the individual stays a real
// vector, decoded into a greedily packed item set only inside
// KnapsackProblem's evaluate(). The instance is built so the optimal packed
// value is known in closed form, letting us check how close the solver gets.
void runKnapsackDemo(AlgorithmType algorithm_type, int n_items, bool show_time);

#endif
