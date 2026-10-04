#include "cec17_problem.h"

CEC17Problem::CEC17Problem(int function_number, int dim)
  : function_number_(function_number), dim_(dim) {}

int CEC17Problem::dimension() const { return dim_; }
double CEC17Problem::lowerBound() const { return -100.0; }
double CEC17Problem::upperBound() const { return 100.0; }

Fitness CEC17Problem::evaluate(const double *x) {
  Fitness f;
  cec17_test_func(const_cast<double *>(x), &f, dim_, 1, function_number_);
  return f;
}

Fitness CEC17Problem::knownOptimum() const {
  return function_number_ * 100;
}
