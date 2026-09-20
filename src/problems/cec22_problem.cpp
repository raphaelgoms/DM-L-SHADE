#include "cec22_problem.h"

namespace {

// Bias added to each of the 12 CEC22 functions (see cec22_test_func.cpp),
// and therefore each function's known optimum error value.
const double kKnownOptimum[] = {
  300.0, 400.0, 600.0, 800.0, 900.0, 1800.0,
  2000.0, 2200.0, 2300.0, 2400.0, 2600.0, 2700.0,
};

}  // namespace

CEC22Problem::CEC22Problem(int function_number, int dim)
  : function_number_(function_number), dim_(dim) {}

int CEC22Problem::dimension() const { return dim_; }
double CEC22Problem::lowerBound() const { return -100.0; }
double CEC22Problem::upperBound() const { return 100.0; }

Fitness CEC22Problem::evaluate(const double *x) {
  Fitness f;
  cec22_test_func(const_cast<double *>(x), &f, dim_, 1, function_number_);
  return f;
}

Fitness CEC22Problem::knownOptimum() const {
  return kKnownOptimum[function_number_ - 1];
}
