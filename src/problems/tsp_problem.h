#ifndef _TSP_PROBLEM_H_
#define _TSP_PROBLEM_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "problem.h"

/*
  Euclidean TSP exposed through the same Problem interface used by continuous
  problems, via random-key encoding: the DE individual stays a plain vector of
  reals in [0, 1), and is only turned into a permutation of cities at
  evaluation time by sorting city indices by their key value (argsort). Since
  any real vector decodes to a valid tour, the solver's mutation/crossover
  operators never need to know they are solving a combinatorial problem.
*/
class TSPProblem : public Problem {
public:
  TSPProblem(std::vector<std::pair<double, double>> cities, Fitness known_optimum_length = -1)
    : cities_(std::move(cities)), known_optimum_(known_optimum_length) {}

  int dimension() const override { return (int)cities_.size(); }
  double lowerBound() const override { return 0.0; }
  double upperBound() const override { return 1.0; }

  Fitness evaluate(const double *x) override {
    return tourLength(decode(x));
  }

  bool hasKnownOptimum() const override { return known_optimum_ >= 0; }
  Fitness knownOptimum() const override { return known_optimum_; }

  // Random-key decoding: argsort the keys to get a permutation of city indices.
  std::vector<int> decode(const double *x) const {
    int n = dimension();
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });
    return order;
  }

  Fitness tourLength(const std::vector<int> &tour) const {
    double total = 0;
    int n = (int)tour.size();
    for (int i = 0; i < n; i++) {
      const auto &a = cities_[tour[i]];
      const auto &b = cities_[tour[(i + 1) % n]];
      double dx = a.first - b.first;
      double dy = a.second - b.second;
      total += std::sqrt(dx * dx + dy * dy);
    }
    return total;
  }

  const std::vector<std::pair<double, double>> &cities() const { return cities_; }

private:
  std::vector<std::pair<double, double>> cities_;
  Fitness known_optimum_;
};

/*
  Cities placed evenly on a circle: for points in convex position the optimal
  TSP tour is known to visit them in cyclic order, so the optimal tour length
  has a closed form (n times the chord length between neighbors). That gives
  a known optimum to check the solver's output against without brute force.
*/
inline std::shared_ptr<TSPProblem> makeCircleTSPInstance(int n_cities, double radius) {
  std::vector<std::pair<double, double>> cities;
  for (int i = 0; i < n_cities; i++) {
    double angle = 2.0 * M_PI * i / n_cities;
    cities.push_back({radius * std::cos(angle), radius * std::sin(angle)});
  }

  double chord = 2.0 * radius * std::sin(M_PI / n_cities);
  double known_optimum = n_cities * chord;
  return std::make_shared<TSPProblem>(cities, known_optimum);
}

#endif
