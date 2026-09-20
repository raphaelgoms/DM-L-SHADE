#include "cec14_problem.h"

CEC14Problem::CEC14Problem(int function_number, int dim)
  : function_number_(function_number), dim_(dim) {}

int CEC14Problem::dimension() const { return dim_; }
double CEC14Problem::lowerBound() const { return -100.0; }
double CEC14Problem::upperBound() const { return 100.0; }

Fitness CEC14Problem::evaluate(const double *x) {
  Fitness f;
  cec14_test_func(const_cast<double *>(x), &f, dim_, 1, function_number_);
  return f;
}

Fitness CEC14Problem::knownOptimum() const {
  return function_number_ * 100;
}
