/*!

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

*/

#pragma once


#include <memory>

#include <pyclustering/cluster/kmedians_data.hpp>

#include <pyclustering/utils/metric.hpp>


using namespace pyclustering::utils::metric;


namespace pyclustering {

namespace clst {


/**
*
* @brief    Represents K-Medians clustering algorithm for cluster analysis.
* @details  The algorithm related to partitional class when input data is divided into groups.
*
*/
class kmedians {
public:
    const static double         DEFAULT_TOLERANCE;  /**< Default value of the tolerance stop condition: if maximum value of change of centers of clusters is less than tolerance then algorithm stops processing. */

    const static std::size_t    DEFAULT_ITERMAX;    /**< Default value of the step stop condition - maximum number of iterations that is used for clustering process. */

private:
    const static double         THRESHOLD_CHANGE;

private:
    double                  m_tolerance         = 0.0;

    std::size_t             m_max_iter          = 0;

    dataset                 m_initial_medians   = { };

    kmedians_data         * m_ptr_result        = nullptr;   /* temporary pointer to output result */

    const dataset         * m_ptr_data          = nullptr;     /* used only during processing */

    distance_metric<point>  m_metric;

public:
    /**
    *
    * @brief    Default constructor of clustering algorithm.
    *
    */
    kmedians() = default;

    /**
    *
    * @brief    Constructor of clustering algorithm where algorithm parameters for processing are
    *           specified.
    *
    * @param[in] p_initial_medians: initial medians that are used for processing.
    * @param[in] p_tolerance: stop condition in following way: when maximum value of distance change of
    *             medians of clusters is less than tolerance than algorithm will stop processing.
    * @param[in] p_max_iter: maximum amount of iteration for clustering.
    * @param[in] p_metric: distance metric for distance calculation between objects.
    *
    */
    kmedians(const dataset & p_initial_medians, 
             const double p_tolerance = DEFAULT_TOLERANCE,
             const std::size_t p_max_iter = DEFAULT_ITERMAX,
             const distance_metric<point> & p_metric = distance_metric_factory<point>::euclidean_square());

    /**
    *
    * @brief    Default destructor of the algorithm.
    *
    */
    ~kmedians() = default;

public:
    /**
    *
    * @brief    Performs cluster analysis of an input data.
    *
    * @param[in]  p_data: input data for cluster analysis.
    * @param[out] p_output_result: clustering result of an input data.
    *
    */
    void process(const dataset & p_data, kmedians_data & p_output_result);

private:
    /**
    *
    * @brief    Updates clusters in line with current medians.
    *
    * @param[in] p_medians: medians that are used for updating clusters.
    * @param[out] p_clusters: updated clusters in line with the specified medians.
    *
    */
    void update_clusters(const dataset & p_medians, cluster_sequence & p_clusters);

    /**
    *
    * @brief    Assign point to cluster by marking corresponding index in container 'p_lables'.
    *
    * @param[in] p_index_point: index of point that should be assigned to cluster.
    * @param[in] p_medians: medians that corresponds to clusters.
    * @param[out] p_lables: cluster labels for each point (cluster labels has the same size as an input data).
    *
    */
    void assign_point_to_cluster(const std::size_t p_index_point, const dataset & p_medians, index_sequence & p_lables);

    /**
    *
    * @brief    Updates medians in line with current clusters.
    *
    * @param[in,out] clusters: clusters that are sorted and used for updating medians.
    * @param[out] medians: updated medians in line with the specified clusters.
    *
    */
    double update_medians(cluster_sequence & clusters, dataset & medians);

    /**
    *
    * @brief    Calculate median for particular cluster.
    *
    * @param[in,out] current_cluster: cluster that is sorted and used for updating medians.
    * @param[out] median: calculate median for particular cluster.
    *
    */
    void calculate_median(cluster & current_cluster, point & median);

    /**
    *
    * @brief    Erases clusters that do not have any points.
    *
    * @param[in,out] p_clusters: clusters that should be analyzed and modified.
    *
    */
    static void erase_empty_clusters(cluster_sequence & p_clusters);
};


}

}