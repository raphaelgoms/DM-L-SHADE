/*
  DM-L-SHADE with local search around the mined patterns, implemented by C++

  A copy of DM-L-SHADE (dm_lshade.cpp) where each pattern (cluster centroid)
  injected into the population is evaluated and then refined by a local
  search: a (1+1)-ES with the 1/5 success rule or Hooke-Jeeves pattern search.
  Its initial step size on each variable is the standard deviation of the
  elite members of that pattern's cluster.

  Written by Raphael Gomes Santos (raphaelgoms [at] gmail.com)
*/

#include "algorithm.h"
#include <algorithm>
#include <pyclustering/cluster/kmeans.hpp>
#include <pyclustering/cluster/xmeans.hpp>
#include <pyclustering/cluster/random_center_initializer.hpp>

using namespace pyclustering;
using namespace pyclustering::clst;

DMLSHADELS::DMLSHADELS(shared_ptr<Problem> problem, const SHADEConfig &config,
                       int max_elite_size, int number_of_patterns, int mining_generation_step,
                       double local_search_rate, LocalSearchMethod local_search_method)
  : searchAlgorithm(problem, config)
{
  this->max_elite_size = max_elite_size;
  this->number_of_patterns = number_of_patterns;
  this->mining_generation_step = mining_generation_step;
  this->local_search_rate = local_search_rate;
  this->local_search_method = local_search_method;
  generation = 0;
}

Fitness DMLSHADELS::run()
{
    cout << scientific << setprecision(8);
    initializeParameters();
    setSHADEParameters();

    vector<Individual> pop;
    vector<Fitness> fitness(pop_size, 0);
    vector<Individual> children;
    vector<Fitness> children_fitness(pop_size, 0);

    // initialize population
    for (int i = 0; i < pop_size; i++)
    {
        pop.push_back(makeNewIndividual());
        children.push_back((variable *)malloc(sizeof(variable) * problem_size));
    }

    // evaluate the initial population's fitness values
    evaluatePopulation(pop, fitness);

    Individual bsf_solution = (variable *)malloc(sizeof(variable) * problem_size);
    Fitness bsf_fitness;
    int nfes = 0;

    if (problem->hasKnownOptimum() && (fitness[0] - optimum) < epsilon)
        fitness[0] = optimum;
    bsf_fitness = fitness[0];
    for (int j = 0; j < problem_size; j++)
        bsf_solution[j] = pop[0][j];
    /////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < pop_size; i++)
    {
        nfes++;

        if (problem->hasKnownOptimum() && (fitness[i] - optimum) < epsilon)
            fitness[i] = optimum;

        if (fitness[i] < bsf_fitness)
        {
            bsf_fitness = fitness[i];
            for (int j = 0; j < problem_size; j++)
                bsf_solution[j] = pop[i][j];
        }

        recordCheckpoints(nfes, bsf_fitness);

        // if (nfes % 1000 == 0) {
        //   //      cout << nfes << " " << bsf_fitness - optimum << endl;
        //   cout << bsf_fitness - optimum << endl;
        // }

        if (shouldStop(nfes, bsf_fitness))
            break;
    }
    ////////////////////////////////////////////////////////////////////////////

    // for external archive
    int arc_ind_count = 0;
    int random_selected_arc_ind;
    vector<Individual> archive;
    for (int i = 0; i < arc_size; i++)
        archive.push_back((variable *)malloc(sizeof(variable) * problem_size));

    int num_success_params;
    vector<variable> success_sf;
    vector<variable> success_cr;
    vector<variable> dif_fitness;

    // the contents of M_f and M_cr are all initialiezed 0.5
    vector<variable> memory_sf(memory_size, 0.5);
    vector<variable> memory_cr(memory_size, 0.5);

    variable temp_sum_sf;
    variable temp_sum_cr;
    variable sum;
    variable weight;

    // memory index counter
    int memory_pos = 0;

    // for new parameters sampling
    variable mu_sf, mu_cr;
    int random_selected_period;
    variable *pop_sf = (variable *)malloc(sizeof(variable) * pop_size);
    variable *pop_cr = (variable *)malloc(sizeof(variable) * pop_size);

    // for current-to-pbest/1
    int p_best_ind;
    int p_num = round(pop_size * p_best_rate);
    int *sorted_array = (int *)malloc(sizeof(int) * pop_size);
    Fitness *temp_fit = (Fitness *)malloc(sizeof(Fitness) * pop_size);

    // for linear population size reduction
    int max_pop_size = pop_size;
    int min_pop_size = 4;
    int plan_pop_size;

    // Patterns set 
    // I'm using a map because the pattern does not necessarily have the same number 
    // of variables that a solution
    vector<map<int, double>> patterns;
    // Standard deviation of each pattern's cluster, per variable.
    vector<vector<double>> pattern_spreads;
    Individual ls_candidate = (variable *)malloc(sizeof(variable) * problem_size);
    int ls_evaluations = max(1, (int)round(local_search_rate * problem_size));

    // main loop
    while (!shouldStop(nfes, bsf_fitness))
    {
        generation++;
        for (int i = 0; i < pop_size; i++)
            sorted_array[i] = i;
        for (int i = 0; i < pop_size; i++)
            temp_fit[i] = fitness[i];
        sortIndexWithQuickSort(&temp_fit[0], 0, pop_size - 1, sorted_array);

        // Mining steps
        updateElite(pop, fitness, sorted_array);
        if (generation % mining_generation_step == 0){
          patterns = minePatterns(pattern_spreads);

          int mpi = min((int)patterns.size(), pop_size); // max patterns insertions
          for (size_t i = 0; i < mpi && !shouldStop(nfes, bsf_fitness); i++) {
            int idx = sorted_array[pop_size - 1 - i];
            for (size_t j = 0; j < problem_size; j++) {
                pop[idx][j] = patterns[i][j];
            }
            localSearch(pop[idx], fitness[idx], pattern_spreads[i], ls_evaluations, ls_candidate,
                        nfes, bsf_solution, bsf_fitness);
          }
          if (shouldStop(nfes, bsf_fitness))
            break;
        }

        for (int target = 0; target < pop_size; target++)
        {
            // In each generation, CR_i and F_i used by each individual x_i are generated by first selecting 
            // an index r_i randomly from [1, H]
            random_selected_period = rand() % memory_size;
            mu_sf = memory_sf[random_selected_period];
            mu_cr = memory_cr[random_selected_period];

            // generate CR_i and repair its value
            if (mu_cr == -1)
            {
                pop_cr[target] = 0;
            }
            else
            {
                pop_cr[target] = gauss(mu_cr, 0.1);
                if (pop_cr[target] > 1)
                    pop_cr[target] = 1;
                else if (pop_cr[target] < 0)
                    pop_cr[target] = 0;
            }

            // generate F_i and repair its value
            do
            {
                pop_sf[target] = cauchy_g(mu_sf, 0.1);
            } while (pop_sf[target] <= 0);

            if (pop_sf[target] > 1)
                pop_sf[target] = 1;

            // p-best individual is randomly selected from the top pop_size *  p_i members
            p_best_ind = sorted_array[rand() % p_num];
            operateCurrentToPBest1BinWithArchive(pop, &children[target][0], target, p_best_ind, pop_sf[target], pop_cr[target], archive, arc_ind_count);
        }

        // evaluate the children's fitness values
        evaluatePopulation(children, children_fitness);

        /////////////////////////////////////////////////////////////////////////
        // update the bsf-solution and check the current number of fitness evaluations
        //  if the current number of fitness evaluations over the max number of fitness evaluations, the search is terminated
        //  So, this program is unconcerned about L-SHADE algorithm directly
        for (int i = 0; i < pop_size; i++)
        {
            nfes++;

            // following the rules of CEC 2014 real parameter competition,
            // if the gap between the error values of the best solution found and the optimal solution was 10^{−8} or smaller,
            // the error was treated as 0
            if (problem->hasKnownOptimum() && (children_fitness[i] - optimum) < epsilon)
                children_fitness[i] = optimum;

            if (children_fitness[i] < bsf_fitness)
            {
                bsf_fitness = children_fitness[i];
                for (int j = 0; j < problem_size; j++)
                    bsf_solution[j] = children[i][j];
            }

            recordCheckpoints(nfes, bsf_fitness);

            // if (nfes % 1000 == 0) {
            // //      cout << nfes << " " << bsf_fitness - optimum << endl;
            // 	cout << bsf_fitness - optimum << endl;
            // }
            if (shouldStop(nfes, bsf_fitness))
                break;
        }
        ////////////////////////////////////////////////////////////////////////////

        // generation alternation
        for (int i = 0; i < pop_size; i++)
        {
            if (children_fitness[i] == fitness[i])
            {
                fitness[i] = children_fitness[i];
                for (int j = 0; j < problem_size; j++)
                    pop[i][j] = children[i][j];
            }
            else if (children_fitness[i] < fitness[i])
            {
                // parent vectors x_i which were worse than the trial vectors u_i are preserved
                if (arc_size > 1)
                {
                    if (arc_ind_count < arc_size)
                    {
                        for (int j = 0; j < problem_size; j++)
                            archive[arc_ind_count][j] = pop[i][j];
                        arc_ind_count++;
                    }
                    // Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted elements
                    else
                    {
                        random_selected_arc_ind = rand() % arc_size;
                        for (int j = 0; j < problem_size; j++)
                            archive[random_selected_arc_ind][j] = pop[i][j];
                    }
                }

                dif_fitness.push_back(fabs(fitness[i] - children_fitness[i]));
                fitness[i] = children_fitness[i];
                for (int j = 0; j < problem_size; j++)
                    pop[i][j] = children[i][j];

                // successful parameters are preserved in S_F and S_CR
                success_sf.push_back(pop_sf[i]);
                success_cr.push_back(pop_cr[i]);
            }
        }

        num_success_params = success_sf.size();

        // if numeber of successful parameters > 0, historical memories are updated
        if (num_success_params > 0)
        {
            memory_sf[memory_pos] = 0;
            memory_cr[memory_pos] = 0;
            temp_sum_sf = 0;
            temp_sum_cr = 0;
            sum = 0;

            for (int i = 0; i < num_success_params; i++)
                sum += dif_fitness[i];

            // weighted lehmer mean
            for (int i = 0; i < num_success_params; i++)
            {
                weight = dif_fitness[i] / sum;

                memory_sf[memory_pos] += weight * success_sf[i] * success_sf[i];
                temp_sum_sf += weight * success_sf[i];

                memory_cr[memory_pos] += weight * success_cr[i] * success_cr[i];
                temp_sum_cr += weight * success_cr[i];
            }

            memory_sf[memory_pos] /= temp_sum_sf;

            if (temp_sum_cr == 0 || memory_cr[memory_pos] == -1)
                memory_cr[memory_pos] = -1;
            else
                memory_cr[memory_pos] /= temp_sum_cr;

            // increment the counter
            memory_pos++;
            if (memory_pos >= memory_size)
                memory_pos = 0;

            // clear out the S_F, S_CR and delta fitness
            success_sf.clear();
            success_cr.clear();
            dif_fitness.clear();
        }

        // calculate the population size in the next generation
        plan_pop_size = plannedPopSize(nfes, min_pop_size, max_pop_size);

        if (pop_size > plan_pop_size)
        {
            reduction_ind_num = pop_size - plan_pop_size;
            if (pop_size - reduction_ind_num < min_pop_size)
                reduction_ind_num = pop_size - min_pop_size;

            reducePopulationWithSort(pop, fitness);

            // resize the archive size
            arc_size = pop_size * arc_rate;
            if (arc_ind_count > arc_size)
                arc_ind_count = arc_size;

            // resize the number of p-best individuals
            p_num = round(pop_size * p_best_rate);
            if (p_num <= 1)
                p_num = 2;
        }
    }

    free(ls_candidate);
    finishRun(nfes, bsf_fitness);
    return bsf_fitness - optimum;
}

// Evaluates a solution counting it like the DE evaluations (bsf, checkpoints).
Fitness DMLSHADELS::evaluateCounted(Individual solution, int &nfes, Individual bsf_solution, Fitness &bsf_fitness)
{
    Fitness f = problem->evaluate(solution);
    nfes++;
    if (problem->hasKnownOptimum() && (f - optimum) < epsilon)
        f = optimum;
    if (f < bsf_fitness)
    {
        bsf_fitness = f;
        for (int j = 0; j < problem_size; j++)
            bsf_solution[j] = solution[j];
    }
    recordCheckpoints(nfes, bsf_fitness);
    return f;
}

// Evaluates x (a pattern just injected) and refines it in place with the
// configured local search, using at most max_evaluations evaluations in all
// and stopping early when the run's stop criterion is met.
void DMLSHADELS::localSearch(Individual x, Fitness &x_fitness, const vector<double> &sigma0,
                             int max_evaluations, Individual candidate, int &nfes,
                             Individual bsf_solution, Fitness &bsf_fitness)
{
    if (local_search_method == LocalSearchMethod::HOOKE_JEEVES)
        hookeJeeves(x, x_fitness, sigma0, max_evaluations, nfes, bsf_solution, bsf_fitness);
    else
        evolutionStrategy(x, x_fitness, sigma0, max_evaluations, candidate, nfes, bsf_solution, bsf_fitness);
}

// (1+1)-ES: each variable is perturbed with a Gaussian step, the candidate
// replaces x when not worse, and the step sizes grow after a success and
// shrink after a failure (1/5 success rule).
void DMLSHADELS::evolutionStrategy(Individual x, Fitness &x_fitness, const vector<double> &sigma0,
                                   int max_evaluations, Individual candidate, int &nfes,
                                   Individual bsf_solution, Fitness &bsf_fitness)
{
    auto evaluate = [&](Individual solution) {
        return evaluateCounted(solution, nfes, bsf_solution, bsf_fitness);
    };

    x_fitness = evaluate(x);

    vector<double> sigma(sigma0);
    for (int e = 1; e < max_evaluations && !shouldStop(nfes, bsf_fitness); e++)
    {
        for (int j = 0; j < problem_size; j++)
            candidate[j] = gauss(x[j], sigma[j]);
        modifySolutionWithParentMedium(candidate, x);

        Fitness candidate_fitness = evaluate(candidate);
        double factor;
        if (candidate_fitness <= x_fitness)
        {
            x_fitness = candidate_fitness;
            for (int j = 0; j < problem_size; j++)
                x[j] = candidate[j];
            factor = exp(1.0 / 3.0);
        }
        else
        {
            factor = exp(-1.0 / 12.0);
        }
        for (int j = 0; j < problem_size; j++)
            sigma[j] *= factor;
    }
}

// Hooke-Jeeves pattern search: an exploratory move tries x_j + delta_j and
// then x_j - delta_j on each variable in turn, keeping each strict improvement.
// When it improves on the base point, a pattern move jumps to
// new + (new - base) and explores around there, repeated while that keeps
// improving; when it fails, every step is halved.
void DMLSHADELS::hookeJeeves(Individual x, Fitness &x_fitness, const vector<double> &delta0,
                             int max_evaluations, int &nfes, Individual bsf_solution, Fitness &bsf_fitness)
{
    int evaluations = 0;
    auto budgetLeft = [&]() {
        return evaluations < max_evaluations && !shouldStop(nfes, bsf_fitness);
    };
    auto evaluate = [&](vector<double> &solution) {
        evaluations++;
        return evaluateCounted(solution.data(), nfes, bsf_solution, bsf_fitness);
    };
    // Exploratory move around point (with fitness f), in place.
    auto explore = [&](vector<double> &point, Fitness &f, const vector<double> &delta) {
        for (int j = 0; j < problem_size && budgetLeft(); j++) {
            double original = point[j];
            for (int direction = 1; direction >= -1 && budgetLeft(); direction -= 2) {
                point[j] = original + direction * delta[j];
                // Out of bounds: halfway between the bound and the original value.
                if (point[j] < min_region) point[j] = (min_region + original) / 2.0;
                if (point[j] > max_region) point[j] = (max_region + original) / 2.0;
                Fitness trial = evaluate(point);
                if (trial < f) {
                    f = trial;
                    break;
                }
                point[j] = original;
            }
        }
    };

    vector<double> base(x, x + problem_size);
    Fitness base_fitness = evaluate(base);
    vector<double> delta(delta0);
    // Stop refining once the steps are negligible.
    double min_delta = 1e-8 * (max_region - min_region);

    while (budgetLeft()) {
        vector<double> current = base;
        Fitness current_fitness = base_fitness;
        explore(current, current_fitness, delta);

        if (current_fitness < base_fitness) {
            // Pattern moves while they keep improving.
            while (true) {
                vector<double> previous = base;
                base = current;
                base_fitness = current_fitness;
                if (!budgetLeft())
                    break;

                vector<double> pattern_point(problem_size);
                for (int j = 0; j < problem_size; j++) {
                    pattern_point[j] = 2 * base[j] - previous[j];
                    if (pattern_point[j] < min_region) pattern_point[j] = (min_region + base[j]) / 2.0;
                    if (pattern_point[j] > max_region) pattern_point[j] = (max_region + base[j]) / 2.0;
                }
                Fitness pattern_fitness = evaluate(pattern_point);
                explore(pattern_point, pattern_fitness, delta);
                if (pattern_fitness >= base_fitness)
                    break;
                current = pattern_point;
                current_fitness = pattern_fitness;
            }
        } else {
            bool all_small = true;
            for (int j = 0; j < problem_size; j++) {
                delta[j] *= 0.5;
                if (delta[j] > min_delta) all_small = false;
            }
            if (all_small)
                break;
        }
    }

    for (int j = 0; j < problem_size; j++)
        x[j] = base[j];
    x_fitness = base_fitness;
}

void DMLSHADELS::operateCurrentToPBest1BinWithArchive(const vector<Individual> &pop, Individual child, int &target, int &p_best_individual, variable &scaling_factor, variable &cross_rate, const vector<Individual> &archive, int &arc_ind_count) {
  int r1, r2;
  
  do {
    r1 = rand() % pop_size;
  } while (r1 == target);
  do {
    r2 = rand() % (pop_size + arc_ind_count);
  } while ((r2 == target) || (r2 == r1));

  int random_variable = rand() % problem_size;
  
  if (r2 >= pop_size) {
    r2 -= pop_size;
    for (int i = 0; i < problem_size; i++) {
      if ((randDouble() < cross_rate) || (i == random_variable)) {
	child[i] = pop[target][i] + scaling_factor * (pop[p_best_individual][i] - pop[target][i]) + scaling_factor * (pop[r1][i] - archive[r2][i]);
      }
      else {
	child[i] = pop[target][i];
      }
    }
  }
  else {
    for (int i = 0; i < problem_size; i++) {
      if ((randDouble() < cross_rate) || (i == random_variable)) {
	child[i] = pop[target][i] + scaling_factor * (pop[p_best_individual][i] - pop[target][i]) + scaling_factor * (pop[r1][i] - pop[r2][i]);
      }
      else {
	child[i] = pop[target][i];
      }
    }
  }

  //If the mutant vector violates bounds, the bound handling method is applied
  modifySolutionWithParentMedium(child,  pop[target]);
}

void DMLSHADELS::updateElite(const vector<Individual> &pop, vector<Fitness> &fitness, int* sorted_indexes)
{
    for (size_t i = 0; i < pop.size(); i++) {
        elite.push_back({ pop[sorted_indexes[i]], fitness[sorted_indexes[i]] });
    }
    std::sort(elite.begin(), elite.end(), [](tuple<Individual, double> el1, tuple<Individual, double> el2) {
      return get<1>(el1) < get<1>(el2);
    });
    
    if (elite.size() > max_elite_size)
        elite.erase(elite.begin() + max_elite_size, elite.end());
}


// Also returns in spreads, for each pattern, the standard deviation of its
// cluster's members per variable (the elite-wide one where the cluster has no
// spread, e.g. a single member), the initial step sizes of the local search.
vector<map<int, double>> DMLSHADELS::minePatterns(vector<vector<double>> &spreads)
{
    vector<vector<double>> data;
    for (size_t i = 0; i < elite.size(); i++)
        data.push_back(vector<double>(get<0>(elite[i]), get<0>(elite[i]) + problem_size));

    vector<vector<double> > start_centers;
    xmeans_data output_result;
    long seed = 1;

    if (number_of_patterns> 0)
	{
		output_result.clusters().clear();
		random_center_initializer(number_of_patterns, seed).initialize(data, start_centers);
		pyclustering::clst::kmeans solver(start_centers);
		solver.process(data, output_result);
	} 
	else 
	{
		random_center_initializer(1, seed).initialize(data, start_centers);
		xmeans solver(start_centers, elite.size(), 1e-4, splitting_type::BAYESIAN_INFORMATION_CRITERION, 1, seed);
		solver.process(data, output_result);
	}
	
    map<int, double> _pattern;
    vector<map<int, double>> patterns;
    auto centers = output_result.centers();
	for(auto center:centers) {
		for (size_t j = 0; j < problem_size; j++) {	
			_pattern.insert(pair<int, double>(j, center[j]));
		}
		patterns.push_back(_pattern);
	}

    auto standardDeviation = [&](const vector<size_t> &members, int j) {
        double mean = 0, var = 0;
        for (size_t m : members) mean += data[m][j];
        mean /= members.size();
        for (size_t m : members) var += (data[m][j] - mean) * (data[m][j] - mean);
        return sqrt(var / members.size());
    };
    vector<size_t> all_members(data.size());
    for (size_t m = 0; m < data.size(); m++) all_members[m] = m;
    // Floor so the search still moves when the whole elite has converged.
    double min_sigma = 1e-8 * (max_region - min_region);

    spreads.clear();
    auto clusters = output_result.clusters();
    for (size_t c = 0; c < centers.size(); c++) {
        vector<double> spread(problem_size);
        for (int j = 0; j < problem_size; j++) {
            double s = c < clusters.size() && !clusters[c].empty() ? standardDeviation(clusters[c], j) : 0;
            if (s <= 0) s = standardDeviation(all_members, j);
            spread[j] = max(s, min_sigma);
        }
        spreads.push_back(spread);
    }

    return patterns;
}

void DMLSHADELS::reducePopulationWithSort(vector<Individual> &pop, vector<Fitness> &fitness) {
  int worst_ind;

  for (int i = 0; i < reduction_ind_num; i++) {
    worst_ind = 0;
    for (int j = 1; j < pop_size; j++) {
      if (fitness[j] > fitness[worst_ind]) worst_ind = j;
    }

    pop.erase(pop.begin() + worst_ind);
    fitness.erase(fitness.begin() + worst_ind);
    pop_size--;
  }
}

void  DMLSHADELS::setSHADEParameters() {
  arc_rate = config.arc_rate;
  arc_size = (int)round(pop_size * arc_rate);
  p_best_rate = config.p_best_rate;
  memory_size = config.memory_size;
}
