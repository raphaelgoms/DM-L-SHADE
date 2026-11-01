#ifndef _CEC26_PROBLEM_H_
#define _CEC26_PROBLEM_H_

#include "cec17_problem.h"

// The bound-constrained single-objective problems of the CEC-2026 competition:
// the CEC-2017 functions (same code and data), numbered 1-29 as the competition
// numbers them (F1 = CEC-2017 f1, F2 = f3, ..., F29 = f30, since f2 was removed).
class CEC26Problem : public CEC17Problem {
public:
  CEC26Problem(int function_number, int dim)
    : CEC17Problem(function_number == 1 ? 1 : function_number + 1, dim) {}
};

#endif
