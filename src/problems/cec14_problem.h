#ifndef _CEC14_PROBLEM_H_
#define _CEC14_PROBLEM_H_

#include "problem.h"

void cec14_test_func(double *, double *, int, int, int);

// Adapts the CEC-2014 benchmark suite to the Problem interface. All of the
// CEC14-specific globals (OShift, M, ini_flag, ...) stay private to
// cec14_test_func.cc; this class only calls into it.
class CEC14Problem : public Problem {
public:
  CEC14Problem(int function_number, int dim);

  int dimension() const override;
  double lowerBound() const override;
  double upperBound() const override;
  Fitness evaluate(const double *x) override;

  bool hasKnownOptimum() const override { return true; }
  Fitness knownOptimum() const override;

private:
  int function_number_;
  int dim_;
};

#endif
