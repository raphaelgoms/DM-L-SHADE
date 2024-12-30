/*!

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

*/

#pragma once


#include <unordered_map>

#include <pyclustering/container/adjacency.hpp>


namespace pyclustering {

namespace container {

/**
*
* @brief   Implementation of adjacency matrix where each node stores its neighbors in unordered
*          set with corresponding weight of connection to corresponding neighbor.
*
* @details Unordered ensures maximum performance in case of getting elements by index because only 
*          indexes are used as keys that ensures uniform distribution. It takes less memory
*          than classical matrix representation. And it is faster in case of getting neighbors than
*          bit matrix and classical matrix representation. Unlike adjacency_list this implementation
*          requires twice as much memory because of storing weight of each connection.
*
* @see     adjacency_bit_matrix
* @see     adjacency_matrix
*
*/
class adjacency_weight_list : public adjacency_weight_collection {
private:
    using adjacency_list_container = std::vector<std::unordered_map<size_t, double>>;


protected:
    adjacency_list_container     m_adjacency;


public:
    /**
    *
    * @brief   Default destructor without arguments is forbiden.
    *
    */
    adjacency_weight_list() = delete;

    /**
    *
    * @brief   Default copy constructor.
    *
    * @param[in]  another_matrix: adjacency matrix that should be copied.
    *
    */
    adjacency_weight_list(const adjacency_weight_list & another_matrix) = default;

    /**
    *
    * @brief   Default move constructor.
    *
    * @param[in]  another_matrix: adjacency matrix that should be moved.
    *
    */
    adjacency_weight_list(adjacency_weight_list && another_matrix) = default;

    /**
    *
    * @brief   Adjacency list constructor.
    *
    * @param[in]  node_amount: number of nodes whose connections are described in matrix.
    *
    */
    explicit adjacency_weight_list(const size_t node_amount);

    /**
    *
    * @brief   Default destructor.
    *
    */
    virtual ~adjacency_weight_list();


private:
    /**
    *
    * @brief   Default value that denotes existance of connection (non-zero weight of connection).
    *
    */
    static const double DEFAULT_EXISTANCE_CONNECTION_VALUE;

    /**
    *
    * @brief   Default value that denotes lack of connection (zero weight of connection).
    *
    */
    static const double DEFAULT_NON_EXISTANCE_CONNECTION_VALUE;


public:
    /**
    *
    * @brief   Returns amount of nodes in the adjacency collection.
    *
    */
    virtual size_t size() const override;

    /**
    *
    * @brief   Establishes one-way connection from the first node to the second in adjacency collection.
    *
    * @details Complexity equals to complexity of insertion of std::unrodered_map.
    *
    * @param[in]  node_index1: index of node in the collection that should be connected with another.
    * @param[in]  node_index2: index of another node in the collection that should be connected with
    *                          the node defined by the first argument 'node_index1'.
    *
    */
    virtual void set_connection(const size_t node_index1, const size_t node_index2) override;

    /**
    *
    * @brief   Removes one-way connection from the first node to the second in adjacency collection.
    *
    * @details Complexity equals to complexity of erasing of std::unrodered_map.
    *
    * @param[in]  node_index1: index of node in the collection that should be disconnected from another.
    * @param[in]  node_index2: index of another node in the collection that should be diconnected from
    *              the node defined by the first argument 'node_index1'.
    *
    */
    virtual void erase_connection(const size_t node_index1, const size_t node_index2) override;

    /**
    *
    * @brief   Checks existance of connection between specified nodes.
    *
    * @details Complexity equal to searching element in std::unrodered_map.
    *
    * @param[in]  node_index1: index of node in the collection.
    * @param[in]  node_index2: index of another node in the collection.
    *
    * @return  'true' - connection between the nodes exists, 'false' - connection does not exist.
    *
    */
    virtual bool has_connection(const size_t node_index1, const size_t node_index2) const override;

    /**
    *
    * @brief   Returns vector of indexes of neighbors of specified node in line with adjacency collection.
    *
    * @details Complexity equals to complexity of traversing of unrodered_map.
    *
    * @param[in]  node_index: index of node in the collection whose neighbors are required.
    * @param[out] node_neighbors: vector of indexes of neighbors of specified node.
    *
    */
    virtual void get_neighbors(const size_t node_index, std::vector<size_t> & node_neighbors) const override;

    /**
    *
    * @brief   Set weight of connection between nodes where zero value means lack of connection and
    *          non-zero means connection with specified weight.
    *
    * @details Complexity equal to searching element in std::unrodered_map.
    *
    * @param[in]  node_index1: index of node in the collection whose connection weight should be updated 
    *              with another node.
    * @param[in]  node_index2: index of another node in the collection.
    * @param[in]  weight: new value of weight of connection between the nodes.
    *
    */
    virtual void set_connection_weight(const size_t node_index1, const size_t node_index2, const double weight) override;

    /**
    *
    * @brief   Returns weight of one-way connection between specified nodes.
    *
    * @details Complexity equal to searching element in std::unrodered_map.
    *
    * @param[in]  node_index1: index of node in the collection whose connection weight should be 
    *              updated with another node.
    * @param[in]  node_index2: index of another node in the collection that is connected to the 
    *              first node.
    *
    * @return  Weight of one-way connection between specified nodes.
    *
    */
    virtual double get_connection_weight(const size_t node_index1, const size_t node_index2) const override;

    /**
    *
    * @brief   Clear content of adjacency matrix.
    *
    */
    virtual void clear() override;

public:
    adjacency_weight_list & operator=(const adjacency_weight_list & another_collection) = default;

    adjacency_weight_list & operator=(adjacency_weight_list && another_collection) = default;
};


}

}