#include "demos/cvrp_demo.h"

#include "problems/cvrp_problem.h"

void runCVRPDemo(AlgorithmType algorithm_type, int n_clusters, int customers_per_cluster, bool show_time) {
  int n_customers = n_clusters * customers_per_cluster;

  cout << "\n-------------------------------------------------------" << endl;
  cout << "CVRP demo (random-key encoding), " << n_clusters << " clusters x "
       << customers_per_cluster << " customers = " << n_customers << " customers\n" << endl;

  auto cvrp = makeClusteredCVRPInstance(n_clusters, customers_per_cluster, 100.0);

  SHADEConfig config;
  config.pop_size = (int)round(n_customers * 18);
  config.max_num_evaluations = n_customers * 10000;
  config.memory_size = 6;
  config.arc_rate = 2.6;
  config.p_best_rate = 0.11;

  searchAlgorithm *alg = createAlgorithm(algorithm_type, cvrp, config);
  Fitness gap_to_optimum = runAlgorithm(alg, show_time);
  delete alg;

  cout << "known optimal total distance = " << cvrp->knownOptimum() << endl;
  cout << "best total distance found    = " << cvrp->knownOptimum() + gap_to_optimum << endl;
  cout << "gap to known optimum         = " << gap_to_optimum << endl;
}
