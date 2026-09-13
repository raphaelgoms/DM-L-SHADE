#ifndef _SET_COVERING_PROBLEM_H_
#define _SET_COVERING_PROBLEM_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "problem.h"

/*
  Set Covering Problem exposed through the same Problem interface used by
  continuous problems, via random-key encoding: the DE individual stays a
  plain vector of reals in [0, 1). The keys are argsorted into a priority
  order over the candidate subsets (same trick as the other combinatorial
  Problems here), which are then greedily picked in that order -- a subset
  is only included if, at that point, it still covers at least one
  uncovered element of the universe (otherwise it's skipped, like
  KnapsackProblem skips items that no longer fit).

  Because the full candidate collection is built (by construction) to cover
  the whole universe, scanning every subset in ANY order always ends with
  full coverage, so every real vector decodes to a feasible cover -- no
  repair or penalty term is needed.
*/
class SetCoveringProblem : public Problem {
public:
  SetCoveringProblem(int universe_size,
                      std::vector<std::vector<int>> subsets,
                      std::vector<double> costs,
                      Fitness known_best_value = -1)
    : universe_size_(universe_size), subsets_(std::move(subsets)),
      costs_(std::move(costs)), known_best_value_(known_best_value) {}

  int dimension() const override { return (int)subsets_.size(); }
  double lowerBound() const override { return 0.0; }
  double upperBound() const override { return 1.0; }

  Fitness evaluate(const double *x) override {
    return coverCost(decode(x));
  }

  bool hasKnownOptimum() const override { return known_best_value_ >= 0; }
  Fitness knownOptimum() const override { return known_best_value_; }

  // Random-key decoding: argsort the keys into a priority order over
  // subsets, then greedily include each subset only while it is still
  // useful (covers at least one element not yet covered).
  std::vector<int> decode(const double *x) const {
    int n = dimension();
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });

    std::vector<bool> covered(universe_size_, false);
    std::vector<int> chosen;
    for (int idx : order) {
      bool useful = false;
      for (int e : subsets_[idx]) {
        if (!covered[e]) { useful = true; break; }
      }

      if (useful) {
        for (int e : subsets_[idx]) covered[e] = true;
        chosen.push_back(idx);
      }
    }
    return chosen;
  }

  Fitness coverCost(const std::vector<int> &chosen) const {
    double total = 0;
    for (int idx : chosen) total += costs_[idx];
    return total;
  }

private:
  int universe_size_;
  std::vector<std::vector<int>> subsets_;
  std::vector<double> costs_;
  Fitness known_best_value_;
};

/*
  Builds an instance out of disjoint blocks of elements. Each block can be
  covered either by one "whole" subset (cost 1) or by two "half" subsets
  covering it together (cost 0.6 + 0.6 = 1.2). Since blocks are disjoint,
  the problem decomposes into independent per-block choices, and covering a
  block by its whole subset (1.0) is always cheaper than via its two halves
  (1.2) -- so the optimal cover uses the whole subset of every block, at a
  known total cost of exactly n_blocks. It's a real trap for the decoder,
  though: a priority order that places both half subsets of a block ahead
  of its whole subset makes the whole subset useless by the time it's
  considered (the block is already fully covered), locking in the more
  expensive 1.2 cost for that block instead.
*/
inline std::shared_ptr<SetCoveringProblem> makeBlockedSetCoveringInstance(int n_blocks, int block_size) {
  int universe_size = n_blocks * block_size;
  std::vector<std::vector<int>> subsets;
  std::vector<double> costs;

  for (int b = 0; b < n_blocks; b++) {
    int start = b * block_size;
    int half = block_size / 2;

    std::vector<int> whole, half1, half2;
    for (int e = start; e < start + block_size; e++) whole.push_back(e);
    for (int e = start; e < start + half; e++) half1.push_back(e);
    for (int e = start + half; e < start + block_size; e++) half2.push_back(e);

    subsets.push_back(whole); costs.push_back(1.0);
    subsets.push_back(half1); costs.push_back(0.6);
    subsets.push_back(half2); costs.push_back(0.6);
  }

  double known_optimum = n_blocks * 1.0;
  return std::make_shared<SetCoveringProblem>(universe_size, subsets, costs, known_optimum);
}

#endif
