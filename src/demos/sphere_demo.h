#ifndef _SPHERE_DEMO_H_
#define _SPHERE_DEMO_H_

#include "algorithm/algorithm_factory.h"

// Demonstrates that the solver is not tied to the CEC14 benchmark: optimizes
// the Sphere function (sum of x_i^2) via a FunctionProblem, never touching
// cec14_test_func.
void runSphereDemo(AlgorithmType algorithm_type, int problem_size, bool show_time);

#endif
