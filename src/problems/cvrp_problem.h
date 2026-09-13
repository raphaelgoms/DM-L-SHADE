#ifndef _CVRP_PROBLEM_H_
#define _CVRP_PROBLEM_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "problem.h"

/*
  Capacitated Vehicle Routing Problem (CVRP) exposed through the same Problem
  interface used by continuous problems, via random-key encoding, extending
  the idea used for TSPProblem: the DE individual stays a plain vector of
  reals in [0, 1). The keys are argsorted into a single "giant tour" over all
  customers (same trick as TSPProblem), which is then greedily cut into
  vehicle routes: whenever the next customer would exceed the vehicle's
  remaining capacity, that route is closed and a new one starts (same
  "skip/cut when it doesn't fit" idea as KnapsackProblem's decoder). An
  unlimited number of vehicles is available, so this always produces a
  feasible set of routes, with no repair or penalty needed.

  The solver only ever sees a flat real vector; it has no notion of routes,
  vehicles or capacity.
*/
class CVRPProblem : public Problem {
public:
  CVRPProblem(std::pair<double, double> depot,
              std::vector<std::pair<double, double>> customers,
              std::vector<double> demands,
              double vehicle_capacity,
              Fitness known_best_value = -1)
    : depot_(depot), customers_(std::move(customers)), demands_(std::move(demands)),
      capacity_(vehicle_capacity), known_best_value_(known_best_value) {}

  int dimension() const override { return (int)customers_.size(); }
  double lowerBound() const override { return 0.0; }
  double upperBound() const override { return 1.0; }

  Fitness evaluate(const double *x) override {
    return totalDistance(decodeRoutes(x));
  }

  bool hasKnownOptimum() const override { return known_best_value_ >= 0; }
  Fitness knownOptimum() const override { return known_best_value_; }

  // Random-key decoding: argsort the keys into a giant tour over customers,
  // then greedily cut it into vehicle routes, starting a new route whenever
  // the next customer would exceed the current route's remaining capacity.
  std::vector<std::vector<int>> decodeRoutes(const double *x) const {
    int n = dimension();
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });

    std::vector<std::vector<int>> routes;
    double load = 0;
    for (int idx : order) {
      if (routes.empty() || load + demands_[idx] > capacity_) {
        routes.push_back({});
        load = 0;
      }
      routes.back().push_back(idx);
      load += demands_[idx];
    }
    return routes;
  }

  Fitness totalDistance(const std::vector<std::vector<int>> &routes) const {
    double total = 0;
    for (const auto &route : routes) {
      std::pair<double, double> prev = depot_;
      for (int idx : route) {
        total += distance(prev, customers_[idx]);
        prev = customers_[idx];
      }
      total += distance(prev, depot_);
    }
    return total;
  }

  static double distance(const std::pair<double, double> &a, const std::pair<double, double> &b) {
    double dx = a.first - b.first;
    double dy = a.second - b.second;
    return std::sqrt(dx * dx + dy * dy);
  }

private:
  std::pair<double, double> depot_;
  std::vector<std::pair<double, double>> customers_;
  std::vector<double> demands_;
  double capacity_;
  Fitness known_best_value_;
};

/*
  Builds an instance made of well-separated clusters, all customers within a
  cluster placed at the same point, each with demand 1, with vehicle capacity
  set to exactly the number of customers per cluster. That capacity match
  means a single vehicle can serve an entire cluster in one visit (using all
  of its capacity), while any other split of the work still needs exactly as
  many fully-loaded vehicles overall -- and, because visiting a cluster costs
  the same regardless of how many of its (colocated) customers are served on
  that visit, spreading a cluster's demand over extra visits or routing a
  vehicle through more than one cluster only adds distance (by the triangle
  inequality) instead of saving any. So dedicating one route per cluster is
  optimal, and its length is known in closed form: for each cluster, twice
  the distance from the depot to that cluster (there and back).
*/
inline std::shared_ptr<CVRPProblem> makeClusteredCVRPInstance(int n_clusters, int customers_per_cluster, double cluster_radius) {
  std::pair<double, double> depot = {0.0, 0.0};
  std::vector<std::pair<double, double>> customers;
  std::vector<double> demands;

  double known_best_value = 0;
  for (int c = 0; c < n_clusters; c++) {
    double angle = 2.0 * M_PI * c / n_clusters;
    std::pair<double, double> center = {cluster_radius * std::cos(angle), cluster_radius * std::sin(angle)};

    for (int j = 0; j < customers_per_cluster; j++) {
      customers.push_back(center);
      demands.push_back(1.0);
    }

    known_best_value += 2.0 * CVRPProblem::distance(depot, center);
  }

  double vehicle_capacity = customers_per_cluster;
  return std::make_shared<CVRPProblem>(depot, customers, demands, vehicle_capacity, known_best_value);
}

#endif
