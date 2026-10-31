#ifndef _CVRPLIB_SOLVER_H_
#define _CVRPLIB_SOLVER_H_

#include <string>

#include "algorithm/algorithm_factory.h"

struct CVRPLibSolverOptions {
  std::string instance_path;   // CVRPLIB .vrp file
  std::string solution_path;   // if not empty, best solution of all runs in CVRPLIB .sol format
  int num_runs = 51;
  bool show_time = false;
  bool record_checkpoints = false;  // requires num_runs == 1
  // CPU-time budget per run in seconds; 0 keeps the evaluation budget, and a
  // negative value means Tmax = n x 240/100 seconds for n customers, the time
  // limit of the protocol of Vidal (2022) for the X instances.
  double cpu_seconds = 0;
  // L-SHADE parameters (--pop-rate, --memory-size, --arc-rate, --p-best-rate);
  // the defaults are those of the CEC runs.
  double pop_rate = 1;  // initial population = pop_rate x customers
  int memory_size = 6;
  double arc_rate = 2.6;
  double p_best_rate = 0.11;
};

// Runs the solver on a CVRPLIB instance (--benchmark cvrplib), with the
// random-key encoding of CVRPProblem, following the conventions of the CEC
// runner: num_runs independent runs with a summary (best, worst, mean, std of
// the total distance and the gap to the best known value when a .sol file
// sits next to the instance). With a single run only the cost is printed, or,
// with record_checkpoints, the best cost at the recording points, so scripts
// can parse it.
//
// Population pop_rate*D with D = number of customers (18*D by default, as in
// the CEC runs). The
// budget is either MaxFES = 10000*D, with costs recorded at 0.01, 0.02, 0.03,
// 0.05, 0.1, ..., 1.0 x MaxFES, or a CPU time (cpu_seconds), with costs
// recorded at 1, 2, 5, 10, 15, 20, 30, 50, 75 and 100% of it as in Vidal (2022).
void runCVRPLibSolver(const AlgorithmSpec &algorithm, const CVRPLibSolverOptions &options);

#endif
