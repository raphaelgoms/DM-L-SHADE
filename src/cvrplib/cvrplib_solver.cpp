#include "cvrplib/cvrplib_solver.h"

#include <climits>
#include <fstream>

#include "problems/cvrplib_instance.h"

namespace {

const double kCheckpointFractions[] = {
  0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
};
// Fractions of the time limit at which Vidal (2022) records the best solution.
const double kTimeCheckpointFractions[] = {0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0};

void writeSolution(const std::string &path, const std::vector<std::vector<int>> &routes, double cost) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  for (size_t r = 0; r < routes.size(); r++) {
    out << "Route #" << r + 1 << ":";
    for (int c : routes[r]) out << " " << c + 1;  // CVRPLIB numbers customers from 1
    out << "\n";
  }
  out << "Cost " << (long long)llround(cost) << "\n";
}

}  // namespace

void runCVRPLibSolver(AlgorithmType algorithm_type, const CVRPLibSolverOptions &options) {
  CVRPLibInstance instance = loadCVRPLibInstance(options.instance_path);
  int n_customers = instance.problem->numCustomers();

  SHADEConfig config;
  config.pop_size = (int)round(n_customers * 18);
  config.max_num_evaluations = n_customers * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;
  if (options.cpu_seconds != 0) {
    config.max_cpu_seconds = options.cpu_seconds > 0 ? options.cpu_seconds : n_customers * 240.0 / 100.0;
    config.max_num_evaluations = UINT_MAX;
  }
  if (options.record_checkpoints) {
    if (config.max_cpu_seconds > 0) {
      for (double fraction : kTimeCheckpointFractions) {
        config.checkpoint_cpu_seconds.push_back(fraction * config.max_cpu_seconds);
      }
    } else {
      for (double fraction : kCheckpointFractions) {
        config.checkpoint_evaluations.push_back((unsigned int)llround(fraction * config.max_num_evaluations));
      }
    }
  }

  // The solver reports values relative to the known optimum when there is one.
  double offset = instance.has_best_known ? instance.best_known : 0.0;
  bool single_run = options.num_runs == 1;

  if (!single_run) {
    cout << "\n-------------------------------------------------------" << endl;
    cout << "Instance = " << instance.name << ", customers = " << n_customers
         << ", capacity = " << instance.problem->capacity();
    if (instance.has_best_known) cout << ", BKS = " << instance.best_known;
    if (config.max_cpu_seconds > 0) cout << ", CPU time limit = " << config.max_cpu_seconds << " s";
    cout << "\n" << endl;
  }

  vector<double> costs;
  vector<std::vector<int>> best_routes;
  double best_cost = 0;
  for (int run = 0; run < options.num_runs; run++) {
    // A fresh copy per run, so the routes it keeps belong to this run only.
    auto problem = std::make_shared<CVRPProblem>(*instance.problem);
    searchAlgorithm *alg = createAlgorithm(algorithm_type, problem, config);
    double cost = offset + runAlgorithm(alg, options.show_time);
    costs.push_back(cost);
    if (run == 0 || cost < best_cost) {
      best_cost = cost;
      best_routes = problem->bestRoutes();
    }

    if (options.record_checkpoints) {
      const vector<Fitness> &errors = alg->checkpointErrors();
      for (size_t k = 0; k < errors.size(); k++) cout << (k ? " " : "") << (long long)llround(offset + errors[k]);
      cout << endl;
    } else if (single_run) {
      cout << (long long)llround(cost) << endl;
    } else {
      cout << run + 1 << "th run, cost = " << (long long)llround(cost);
      if (instance.has_best_known) {
        cout << ", gap = " << fixed << setprecision(2)
             << 100.0 * (cost - instance.best_known) / instance.best_known << "%" << defaultfloat;
      }
      cout << ", routes = " << problem->bestRoutes().size() << endl;
    }
    delete alg;
  }

  if (!single_run) {
    double mean = 0, std_dev = 0;
    for (double c : costs) mean += c;
    mean /= costs.size();
    for (double c : costs) std_dev += (c - mean) * (c - mean);
    std_dev = sqrt(std_dev / costs.size());
    double worst = *max_element(costs.begin(), costs.end());

    cout << fixed << setprecision(2);
    cout << "\nbest = " << best_cost << ", worst = " << worst << ", mean = " << mean << ", std = " << std_dev << endl;
    if (instance.has_best_known) {
      auto gap = [&](double c) { return 100.0 * (c - instance.best_known) / instance.best_known; };
      cout << "gap to BKS: best = " << gap(best_cost) << "%, mean = " << gap(mean) << "%" << endl;
    }
    cout << defaultfloat;
  }

  if (!options.solution_path.empty()) writeSolution(options.solution_path, best_routes, best_cost);
}
