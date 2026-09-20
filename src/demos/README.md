# Demos

Each demo wires one `Problem` instance and a `SHADEConfig` through `createAlgorithm` / `runAlgorithm`. They show that the solver is not tied to the CEC benchmarks, and they are the template to follow when adding a new problem.

| Demo | Flag | Extra parameters | Encoding |
| --- | --- | --- | --- |
| [sphere_demo.cpp](sphere_demo.cpp) | `--sphere` | | continuous |
| [tsp_demo.cpp](tsp_demo.cpp) | `--tsp` | `--tsp-cities N` | random-key |
| [knapsack_demo.cpp](knapsack_demo.cpp) | `--knapsack` | `--knapsack-items N` | random-key |
| [cvrp_demo.cpp](cvrp_demo.cpp) | `--cvrp` | `--cvrp-clusters N`, `--cvrp-cluster-size N` | random-key |
| [set_covering_demo.cpp](set_covering_demo.cpp) | `--setcover` | `--setcover-blocks N`, `--setcover-block-size N` | random-key |

All of them accept `--algorithm lshade|dmlshade` and `--show-time`. Run from the repository root:

```
make
./solver --tsp --tsp-cities 20 --algorithm dmlshade
```

## Adding a new problem

The solver only depends on the abstract `Problem` interface ([problem.h](../problems/problem.h)), so a new problem never requires changes to `searchAlgorithm`, `LSHADE` or `DMLSHADE`. Adding one takes three pieces: the problem class, a demo, and a CLI flag in [main.cpp](../main.cpp). The `Makefile` globs `src/**/*.cpp`, so new files are picked up automatically.

Two examples follow: a continuous problem written from scratch (Rastrigin), and a combinatorial one using the existing TSP demo as a walkthrough.

## Example 1: a continuous problem (Rastrigin)

### 1. Implement `Problem`

Create `src/problems/rastrigin_problem.h`:

```cpp
#ifndef _RASTRIGIN_PROBLEM_H_
#define _RASTRIGIN_PROBLEM_H_

#include <cmath>

#include "problems/problem.h"

class RastriginProblem : public Problem {
public:
  explicit RastriginProblem(int dim) : dim_(dim) {}

  int dimension() const override { return dim_; }
  double lowerBound() const override { return -5.12; }
  double upperBound() const override { return 5.12; }

  // The solver MINIMIZES. To maximize something, return its negative.
  Fitness evaluate(const double *x) override {
    double sum = 10.0 * dim_;
    for (int i = 0; i < dim_; i++)
      sum += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
    return sum;
  }

  // Optional: lets the demo report a gap to the optimum.
  bool hasKnownOptimum() const override { return true; }
  Fitness knownOptimum() const override { return 0.0; }

private:
  int dim_;
};

#endif
```

- `lowerBound()` / `upperBound()` are a single box applied to **every** dimension.
- `evaluate()` receives `dimension()` values and must return a finite value for any vector inside the bounds.
- If `hasKnownOptimum()` is `true`, `run()` returns the **error** (`best fitness - knownOptimum()`) instead of the raw fitness, and errors below the solver's tolerance are snapped to zero. Otherwise `run()` returns the raw best fitness.
- For a stateless function you don't need a subclass: wrap a lambda in `FunctionProblem` (see [sphere_demo.cpp](sphere_demo.cpp)).

### 2. Add the demo

Create `src/demos/rastrigin_demo.h`:

```cpp
#ifndef _RASTRIGIN_DEMO_H_
#define _RASTRIGIN_DEMO_H_

#include "algorithm/algorithm_factory.h"

void runRastriginDemo(AlgorithmType algorithm_type, int problem_size, bool show_time);

#endif
```

and `src/demos/rastrigin_demo.cpp`:

```cpp
#include "demos/rastrigin_demo.h"

#include "problems/rastrigin_problem.h"

void runRastriginDemo(AlgorithmType algorithm_type, int problem_size, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "Rastrigin demo, Dimension size = " << problem_size << "\n" << endl;

  auto problem = make_shared<RastriginProblem>(problem_size);

  SHADEConfig config;
  config.pop_size = (int)round(problem_size * 18);
  config.max_num_evaluations = problem_size * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, problem, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "gap to known optimum = " << gap_to_optimum << endl;
}
```

`SHADEConfig` holds the algorithm hyperparameters (population size, evaluation budget, archive rate, p-best rate, memory size). The values above are the ones used by the other demos; tune them per problem if needed.

### 3. Wire a CLI flag

In [main.cpp](../main.cpp):

```cpp
#include "demos/rastrigin_demo.h"          // with the other demo includes

bool rastrigin_demo = false;               // with the other demo flags in main()

} else if (strcmp(argv[i], "--rastrigin") == 0) {   // in the argument loop
  rastrigin_demo = true;

if (rastrigin_demo) {                      // next to the other `if (..._demo)` blocks
  runRastriginDemo(algorithm_type, problem_size, show_time);
  return 0;
}
```

### 4. Build and run

```
make
./solver --rastrigin
./solver --rastrigin --algorithm dmlshade
```

## Example 2: a combinatorial problem (TSP)

The DE individual is always a real vector, so a combinatorial problem is solved with **random-key encoding**: the problem declares bounds `[0, 1]` and turns the vector into a discrete solution inside `evaluate()`. The solver's operators never know the problem is combinatorial.

This example walks through the existing TSP implementation ([tsp_problem.h](../problems/tsp_problem.h), [tsp_demo.cpp](tsp_demo.cpp)); [knapsack_problem.h](../problems/knapsack_problem.h), [cvrp_problem.h](../problems/cvrp_problem.h) and [set_covering_problem.h](../problems/set_covering_problem.h) follow the same structure.

### 1. Implement `Problem` with a decoder

```cpp
class TSPProblem : public Problem {
public:
  TSPProblem(std::vector<std::pair<double, double>> cities, Fitness known_optimum_length = -1)
    : cities_(std::move(cities)), known_optimum_(known_optimum_length) {}

  // One key per city, all in [0, 1).
  int dimension() const override { return (int)cities_.size(); }
  double lowerBound() const override { return 0.0; }
  double upperBound() const override { return 1.0; }

  Fitness evaluate(const double *x) override {
    return tourLength(decode(x));
  }

  bool hasKnownOptimum() const override { return known_optimum_ >= 0; }
  Fitness knownOptimum() const override { return known_optimum_; }

  // Random-key decoding: argsort the keys to get a permutation of city indices.
  std::vector<int> decode(const double *x) const {
    int n = dimension();
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });
    return order;
  }

  Fitness tourLength(const std::vector<int> &tour) const { /* sum of Euclidean edge lengths */ }

private:
  std::vector<std::pair<double, double>> cities_;
  Fitness known_optimum_;
};
```

Points to follow when writing your own:

- **`decode()` must always produce a feasible solution.** Any real vector in `[0, 1)` argsorts into a valid permutation, so there is nothing to repair or penalize. For constrained problems, build feasibility into the decoder instead: the knapsack decoder walks items in key order and skips those that don't fit, and the CVRP decoder does the equivalent for vehicle capacity.
- **Keep the decoder separate from the objective** (`decode()` vs `tourLength()`), so the decoded solution can be inspected and printed by the demo.
- **Minimize.** If the natural objective is a maximization, return its negative from `evaluate()`. `KnapsackProblem` does this (`return -packedValue(decode(x))`) and negates the known optimum too.

### 2. Provide an instance with a known optimum

Rather than only wrapping user data, each problem header exposes a `make*Instance()` helper that builds an instance whose optimum is known by construction, so the demo can print a gap instead of a bare fitness value. For TSP, cities are placed evenly on a circle, where the optimal tour is the cyclic order and its length has a closed form:

```cpp
inline std::shared_ptr<TSPProblem> makeCircleTSPInstance(int n_cities, double radius) {
  std::vector<std::pair<double, double>> cities;
  for (int i = 0; i < n_cities; i++) {
    double angle = 2.0 * M_PI * i / n_cities;
    cities.push_back({radius * std::cos(angle), radius * std::sin(angle)});
  }

  double chord = 2.0 * radius * std::sin(M_PI / n_cities);
  double known_optimum = n_cities * chord;
  return std::make_shared<TSPProblem>(cities, known_optimum);
}
```

If you can't build such an instance, leave `hasKnownOptimum()` as `false`; `run()` then returns the raw best fitness.

### 3. Add the demo

The demo builds the instance, sets the budget proportional to the number of decision variables, and converts the returned gap back into a domain value:

```cpp
void runTSPDemo(AlgorithmType algorithm_type, int n_cities, bool show_time) {
  cout << "\n-------------------------------------------------------" << endl;
  cout << "TSP demo (random-key encoding), number of cities = " << n_cities << "\n" << endl;

  auto tsp = makeCircleTSPInstance(n_cities, 100.0);

  SHADEConfig config;
  config.pop_size = (int)round(n_cities * 18);
  config.max_num_evaluations = n_cities * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, tsp, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "known optimal tour length = " << tsp->knownOptimum() << endl;
  cout << "best tour length found    = " << tsp->knownOptimum() + gap_to_optimum << endl;
  cout << "gap to known optimum      = " << gap_to_optimum << endl;
}
```

Because `hasKnownOptimum()` is `true`, `runAlgorithm` returns `best - knownOptimum()`, so the best tour length is `knownOptimum() + gap_to_optimum`. For a maximization problem stored as a negated objective (knapsack), the best value is `-knownOptimum() - gap_to_optimum`; see [knapsack_demo.cpp](knapsack_demo.cpp).

### 4. Wire the CLI flags

A combinatorial demo usually takes an instance-size parameter. In [main.cpp](../main.cpp), parse and validate it in the argument loop, then dispatch:

```cpp
bool tsp_demo = false;                     // with the other demo flags in main()
int tsp_n_cities = 15;

} else if (strcmp(argv[i], "--tsp") == 0) {                     // in the argument loop
  tsp_demo = true;
} else if (strcmp(argv[i], "--tsp-cities") == 0 && i + 1 < argc) {
  tsp_n_cities = atoi(argv[i + 1]);
  if (tsp_n_cities < 3) {
    cerr << "Invalid number of cities. Please use a value >= 3." << endl;
    return 1;
  }
  i++;

if (tsp_demo) {                            // next to the other `if (..._demo)` blocks
  runTSPDemo(algorithm_type, tsp_n_cities, show_time);
  return 0;
}
```

```
make
./solver --tsp --tsp-cities 20
./solver --tsp --tsp-cities 20 --algorithm dmlshade
```
