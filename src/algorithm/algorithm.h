/*
  L-SHADE implemented by C++ for Special Session & Competition on Real-Parameter Single Objective Optimization at CEC-2014

  Version: 1.1  Date: 9/Jun/2014
  Written by Ryoji Tanabe (rt.ryoji.tanabe [at] gmail.com)
*/

#ifndef _ALGORITHM_H_
#define _ALGORITHM_H_

#include <stdlib.h>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>
#include <math.h>
#include <ctime>

#include "problems/problem.h"

using namespace std;

typedef  double variable;
typedef variable *Individual;
typedef map<int, double> pattern;

// Hyperparameters of the (L/DM-L)SHADE algorithm itself, independent of the
// problem being optimized.
struct SHADEConfig {
  int pop_size;
  unsigned int max_num_evaluations;
  double arc_rate;
  double p_best_rate;
  int memory_size;
  // Evaluation counts (ascending) at which run() records the best error
  // found so far (see checkpointErrors()). Empty means no recording.
  vector<unsigned int> checkpoint_evaluations;
  // Stop as soon as the error drops below 1e-8 (CEC-2022 rules) instead of
  // always using the whole budget. Needs a problem with a known optimum.
  bool stop_at_optimum = false;
  // CPU-time budget in seconds (e.g. the CVRP protocol of Vidal, 2022). When
  // > 0 the run stops at this CPU time instead of after max_num_evaluations,
  // and the population size reduction follows the elapsed fraction of it.
  double max_cpu_seconds = 0;
  // CPU times in seconds (ascending) at which run() records the best error;
  // used instead of checkpoint_evaluations when not empty.
  vector<double> checkpoint_cpu_seconds;
};

class searchAlgorithm {
public:
  searchAlgorithm(shared_ptr<Problem> problem, const SHADEConfig &config)
    : problem(problem), config(config) {}
  virtual ~searchAlgorithm() {}

  virtual Fitness run() = 0;

  // Best error (best fitness - known optimum) after each of
  // config.checkpoint_evaluations (or config.checkpoint_cpu_seconds), filled
  // by run(). Checkpoints after an early stop hold the final error.
  const vector<Fitness> &checkpointErrors() const { return checkpoint_errors; }
  // Number of evaluations used when run() stopped (FE_term in CEC-2022).
  unsigned int terminationEvaluations() const { return termination_evaluations; }
protected:
  void evaluatePopulation(const vector<Individual> &pop, vector<Fitness> &fitness);

  void initializeParameters();
  void recordCheckpoints(unsigned int nfes, Fitness bsf_fitness);
  bool shouldStop(unsigned int nfes, Fitness bsf_fitness) const;
  void finishRun(unsigned int nfes, Fitness bsf_fitness);
  // CPU seconds since run() started, read from the clock at most every few
  // evaluations (a clock read costs about as much as a cheap evaluation).
  double cpuSeconds(unsigned int nfes) const;
  // Population size planned by the linear reduction after nfes evaluations.
  int plannedPopSize(unsigned int nfes, int min_pop_size, int max_pop_size) const;
  Individual makeNewIndividual();
  void modifySolutionWithParentMedium(Individual child, Individual parent);
  void setBestSolution(const vector<Individual> &pop, const vector<Fitness> &fitness, Individual &bsf_solution, Fitness &bsf_fitness);

  //Return random value with uniform distribution [0, 1)
  inline double randDouble() {
    return (double)rand() / (double) RAND_MAX;
  }

  /*
    Return random value from Cauchy distribution with mean "mu" and variance "gamma"
    http://www.sat.t.u-tokyo.ac.jp/~omi/random_variables_generation.html#Cauchy
  */
  inline double cauchy_g(double mu, double gamma) {
    return mu + gamma * tan(M_PI*(randDouble() - 0.5));
  }

  /*
    Return random value from normal distribution with mean "mu" and variance "gamma"
    http://www.sat.t.u-tokyo.ac.jp/~omi/random_variables_generation.html#Gauss
  */
  inline double gauss(double mu, double sigma){
    return mu + sigma * sqrt(-2.0 * log(randDouble())) * sin(2.0 * M_PI * randDouble());
  }

  //Recursive quick sort with index array
  template<class VarType>
    void sortIndexWithQuickSort(VarType array[], int first, int last, int index[]) {
    VarType x = array[(first + last) / 2];
    int i = first;
    int j = last;
    VarType temp_var = 0;
    int temp_num = 0;

    while (true) {
      while (array[i] < x) i++;    
      while (x < array[j]) j--;      
      if (i >= j) break;

      temp_var = array[i];
      array[i] = array[j];
      array[j] = temp_var;

      temp_num = index[i];
      index[i] = index[j];
      index[j] = temp_num;

      i++;
      j--;
    }

    if (first < (i -1)) sortIndexWithQuickSort(array, first, i - 1, index);  
    if ((j + 1) < last) sortIndexWithQuickSort(array, j + 1, last, index);    
  }
  
  shared_ptr<Problem> problem;
  SHADEConfig config;

  int problem_size;
  variable max_region;
  variable min_region;
  Fitness optimum;
  // acceptable error value
  Fitness epsilon;
  unsigned int max_num_evaluations;
  int pop_size;

  // Best errors recorded so far at config.checkpoint_evaluations.
  vector<Fitness> checkpoint_errors;
  unsigned int termination_evaluations;
  std::clock_t start_clock;
  mutable double cached_cpu_seconds;
  mutable unsigned int cached_cpu_nfes;
};

class LSHADE: public searchAlgorithm {
public:
  LSHADE(shared_ptr<Problem> problem, const SHADEConfig &config)
    : searchAlgorithm(problem, config) {}

  virtual Fitness run();
  void setSHADEParameters();
  void reducePopulationWithSort(vector<Individual> &pop, vector<Fitness> &fitness);
  void operateCurrentToPBest1BinWithArchive(const vector<Individual> &pop, Individual child, int &target, int &p_best_individual, variable &scaling_factor, variable &cross_rate, const vector<Individual> &archive, int &arc_ind_count);

  int arc_size;
  double arc_rate;
  variable p_best_rate;
  int memory_size;
  int reduction_ind_num;
};

class DMLSHADE: public searchAlgorithm {
public:
  DMLSHADE(shared_ptr<Problem> problem, const SHADEConfig &config,
           int max_elite_size, int number_of_patterns, int mining_generation_step);

  virtual Fitness run();
  void setSHADEParameters();
  void reducePopulationWithSort(vector<Individual> &pop, vector<Fitness> &fitness);
  void operateCurrentToPBest1BinWithArchive(const vector<Individual> &pop, Individual child, int &target, int &p_best_individual, variable &scaling_factor, variable &cross_rate, const vector<Individual> &archive, int &arc_ind_count);

  void updateElite(const vector<Individual> &pop, vector<Fitness> &fitness, int* sorted_indexes);
  vector<map<int, double>> minePatterns();
  
  int arc_size;
  double arc_rate;
  variable p_best_rate;
  int memory_size;
  int reduction_ind_num;

  int generation;
  int max_elite_size;
  int number_of_patterns;
  int mining_generation_step;

  vector<tuple<Individual, double>> elite;
};

#endif


