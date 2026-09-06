#ifndef _PROBLEM_H_
#define _PROBLEM_H_

#include <functional>

typedef double Fitness;

// Generic continuous optimization problem: dimension, box bounds and an
// objective function. LSHADE/DMLSHADE depend only on this interface, so any
// problem that implements it can be optimized without touching the solver.
class Problem {
public:
  virtual ~Problem() {}

  virtual int dimension() const = 0;
  virtual double lowerBound() const = 0;
  virtual double upperBound() const = 0;
  virtual Fitness evaluate(const double *x) = 0;

  // Only meaningful for benchmarks (like CEC14) that define an error value
  // relative to a known optimum and round it to zero within a tolerance.
  virtual bool hasKnownOptimum() const { return false; }
  virtual Fitness knownOptimum() const { return 0; }
};

// Wraps a plain function so ad-hoc problems don't require writing a Problem
// subclass.
class FunctionProblem : public Problem {
public:
  FunctionProblem(int dim, double lower, double upper,
                  std::function<double(const double *, int)> fn)
    : dim_(dim), lower_(lower), upper_(upper), fn_(fn) {}

  int dimension() const override { return dim_; }
  double lowerBound() const override { return lower_; }
  double upperBound() const override { return upper_; }
  Fitness evaluate(const double *x) override { return fn_(x, dim_); }

private:
  int dim_;
  double lower_, upper_;
  std::function<double(const double *, int)> fn_;
};

#endif
