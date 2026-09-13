#ifndef _KNAPSACK_PROBLEM_H_
#define _KNAPSACK_PROBLEM_H_

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include "problem.h"

/*
  0/1 Knapsack exposed through the same Problem interface used by continuous
  problems, via random-key encoding: the DE individual stays a plain vector of
  reals in [0, 1). It is decoded by treating the keys as a priority order
  (argsort, same trick as TSPProblem) and then greedily packing items in that
  order while they still fit. Skipping items that don't fit means every real
  vector decodes to a feasible knapsack solution, so no repair or penalty
  term is needed to keep the solver inside the capacity constraint.

  The solver minimizes, so evaluate() returns the negative of the packed
  value.
*/
class KnapsackProblem : public Problem {
public:
  KnapsackProblem(std::vector<double> weights, std::vector<double> values,
                   double capacity, Fitness known_best_value = -1)
    : weights_(std::move(weights)), values_(std::move(values)),
      capacity_(capacity), known_best_value_(known_best_value) {}

  int dimension() const override { return (int)weights_.size(); }
  double lowerBound() const override { return 0.0; }
  double upperBound() const override { return 1.0; }

  Fitness evaluate(const double *x) override {
    return -packedValue(decode(x));
  }

  bool hasKnownOptimum() const override { return known_best_value_ >= 0; }
  Fitness knownOptimum() const override { return -known_best_value_; }

  // Random-key decoding: argsort the keys into a priority order over items.
  std::vector<int> decode(const double *x) const {
    int n = dimension();
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });
    return order;
  }

  // Greedily packs items in the given priority order, skipping any that no
  // longer fit. Always feasible by construction.
  double packedValue(const std::vector<int> &order) const {
    double remaining = capacity_;
    double total_value = 0;
    for (int idx : order) {
      if (weights_[idx] <= remaining) {
        remaining -= weights_[idx];
        total_value += values_[idx];
      }
    }
    return total_value;
  }

  double capacity() const { return capacity_; }

private:
  std::vector<double> weights_;
  std::vector<double> values_;
  double capacity_;
  Fitness known_best_value_;
};

/*
  Builds a random instance where the optimal value is known in closed form:
  weights are random, values are set equal to weights (so packing is exactly
  a subset-sum problem), and the capacity is set to the total weight of a
  randomly planted subset. Since every feasible subset's value (== its
  weight sum) can be at most the capacity, and the planted subset attains
  that capacity exactly, the planted subset is provably optimal.
*/
inline std::shared_ptr<KnapsackProblem> makeSubsetSumKnapsackInstance(int n_items) {
  std::vector<double> weights(n_items);
  for (auto &w : weights) w = 1 + (rand() % 50);

  std::vector<int> planted;
  for (int i = 0; i < n_items; i++) {
    if (rand() % 2 == 0) planted.push_back(i);
  }
  if (planted.empty()) planted.push_back(rand() % n_items);

  double capacity = 0;
  for (int i : planted) capacity += weights[i];

  std::vector<double> values = weights;

  return std::make_shared<KnapsackProblem>(weights, values, capacity, capacity);
}

#endif
