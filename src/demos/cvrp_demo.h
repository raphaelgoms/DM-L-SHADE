#ifndef _CVRP_DEMO_H_
#define _CVRP_DEMO_H_

#include "algorithm/algorithm_factory.h"

// Demonstrates applying the solver to a third discrete/combinatorial problem
// (Capacitated Vehicle Routing) via random-key encoding: the individual
// stays a real vector, decoded into a giant tour that is greedily cut into
// vehicle routes only inside CVRPProblem's evaluate(). The instance is built
// out of well-separated, capacity-matched customer clusters so the optimal
// total distance (one dedicated route per cluster) is known in closed form.
void runCVRPDemo(AlgorithmType algorithm_type, int n_clusters, int customers_per_cluster, bool show_time);

#endif
