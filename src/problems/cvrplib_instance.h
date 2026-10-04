#ifndef _CVRPLIB_INSTANCE_H_
#define _CVRPLIB_INSTANCE_H_

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cvrp_problem.h"

/*
  Reads a CVRP instance in the CVRPLIB/TSPLIB format (.vrp), e.g. the X set of
  Uchoa et al. (2017): NAME, DIMENSION, CAPACITY and EDGE_WEIGHT_TYPE header
  fields, then NODE_COORD_SECTION, DEMAND_SECTION and DEPOT_SECTION. Only
  EUC_2D instances are supported; their distances are rounded to the nearest
  integer, as TSPLIB defines EUC_2D and as the published best known solutions
  are computed.

  If a CVRPLIB solution file with the same name and a .sol extension sits
  next to the instance, its "Cost" line is taken as the best known value, so
  runs can report the gap to it.
*/
struct CVRPLibInstance {
  std::string name;
  std::shared_ptr<CVRPProblem> problem;
  bool has_best_known = false;
  double best_known = 0;
};

inline CVRPLibInstance loadCVRPLibInstance(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  CVRPLibInstance instance;
  int dimension = 0;
  double capacity = 0;
  std::string edge_weight_type;
  std::map<int, std::pair<double, double>> coords;
  std::map<int, double> demands;
  std::vector<int> depots;

  std::string line, section;
  while (std::getline(in, line)) {
    std::istringstream tokens(line);
    std::string first;
    if (!(tokens >> first)) continue;

    if (first == "NODE_COORD_SECTION" || first == "DEMAND_SECTION" || first == "DEPOT_SECTION") {
      section = first;
      continue;
    }
    if (first == "EOF") break;

    size_t colon = line.find(':');
    if (colon != std::string::npos) {  // "KEY : value" header field
      std::string key = line.substr(0, colon), value = line.substr(colon + 1);
      auto trim = [](std::string s) {
        size_t b = s.find_first_not_of(" \t\r\""), e = s.find_last_not_of(" \t\r\"");
        return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
      };
      key = trim(key);
      value = trim(value);
      if (key == "NAME") instance.name = value;
      else if (key == "DIMENSION") dimension = std::stoi(value);
      else if (key == "CAPACITY") capacity = std::stod(value);
      else if (key == "EDGE_WEIGHT_TYPE") edge_weight_type = value;
      section.clear();
      continue;
    }

    int id = std::stoi(first);
    if (section == "NODE_COORD_SECTION") {
      double x, y;
      tokens >> x >> y;
      coords[id] = {x, y};
    } else if (section == "DEMAND_SECTION") {
      double d;
      tokens >> d;
      demands[id] = d;
    } else if (section == "DEPOT_SECTION") {
      if (id == -1) section.clear();
      else depots.push_back(id);
    }
  }

  if (edge_weight_type != "EUC_2D") {
    throw std::runtime_error(path + ": EDGE_WEIGHT_TYPE " + edge_weight_type + " is not supported (only EUC_2D)");
  }
  if (depots.size() != 1) throw std::runtime_error(path + ": expected exactly one depot");
  if ((int)coords.size() != dimension || (int)demands.size() != dimension) {
    throw std::runtime_error(path + ": NODE_COORD_SECTION or DEMAND_SECTION does not match DIMENSION");
  }

  // Customers keep the order of their node ids, so customer i is the i-th
  // non-depot node -- the numbering CVRPLIB .sol files use.
  std::pair<double, double> depot = coords[depots[0]];
  std::vector<std::pair<double, double>> customers;
  std::vector<double> customer_demands;
  for (const auto &node : coords) {
    if (node.first == depots[0]) continue;
    customers.push_back(node.second);
    customer_demands.push_back(demands[node.first]);
  }

  std::string sol_path = path.substr(0, path.find_last_of('.')) + ".sol";
  std::ifstream sol(sol_path);
  while (sol && std::getline(sol, line)) {
    std::istringstream tokens(line);
    std::string key;
    if (tokens >> key && key == "Cost" && tokens >> instance.best_known) instance.has_best_known = true;
  }

  instance.problem = std::make_shared<CVRPProblem>(depot, customers, customer_demands, capacity,
                                                   instance.has_best_known ? instance.best_known : -1,
                                                   /*round_distances=*/true);
  return instance;
}

#endif
