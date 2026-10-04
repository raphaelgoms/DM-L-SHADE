#ifndef _CEC17_PROBLEM_H_
#define _CEC17_PROBLEM_H_

#include "problem.h"

void cec17_test_func(double *, double *, int, int, int);

// Adapts the CEC-2017 benchmark suite to the Problem interface. All of the
// CEC17-specific globals (OShift, M, ini_flag, ...) stay private to
// cec17_test_func.cpp; this class only calls into it. Function numbers follow
// the official code (1-30, F*_i = 100*i); F2 was removed from the suite and
// is not a valid function number.
class CEC17Problem : public Problem {
public:
  CEC17Problem(int function_number, int dim);

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
