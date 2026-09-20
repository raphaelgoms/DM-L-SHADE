#ifndef _CEC22_PROBLEM_H_
#define _CEC22_PROBLEM_H_

#include "problem.h"

void cec22_test_func(double *, double *, int, int, int);

// Adapts the CEC-2022 benchmark suite to the Problem interface. All of the
// CEC22-specific globals (OShift, M, ini_flag, ...) stay private to
// cec22_test_func.cpp; this class only calls into it.
class CEC22Problem : public Problem {
public:
  CEC22Problem(int function_number, int dim);

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
