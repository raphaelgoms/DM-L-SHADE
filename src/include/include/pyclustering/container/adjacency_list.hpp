/*!

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

*/


#pragma once


#include <unordered_set>
#include <pyclustering/container/adjacency.hpp>


namespace pyclustering {

namespace container {

/**
*
* @brief   Implementation of adjacency matrix where each node stores its neighbors in unordered
*          set.
*
* @details Unordered ensures maximum performance in case of getting elements by index because only 
*          indexes are used as keys that ensures uniform distribution. It takes less memory
*          than classical matrix representation. And it is faster in case of getting neighbors than
*          bit matrix and classical matrix representation.
*
* @see     adjacency_bit_matrix
* @see     adjacency_matrix
*
*/
class adjacency_list : public adjacency_collection {
private:
    typedef std::vector<std::unordered_set<size_t>>     adjacency_list_container;


protected:
    adjacency_list_container     m_adjacency;


public:
    /**
    *
    * @brief   Default destructor.
    *
    */
    adjacency_list();

    /**
    *
    * @brief   Default copy constructor.
    *
    * @param[in]  another_matrix: adjacency matrix that should be copied.
    *
    */
    adjacency_list(const adjacency_list & another_matrix) = default;

    /**
    *
    * @brief   Default move constructor.
    *
    * @param[in]  another_matrix: adjacency matrix that should be moved.
    *
    */
    adjacency_list(adjacency_list && another_matrix) = default;

    /**
    *
    * @brief   Adjacency list matrix constructor.
    *
    * @param[in]  node_amount: number of nodes whose connections are described in matrix.
    *
    */
    explicit adjacency_list(const size_t node_amount);

    /**
    *
    * @brief   Default destructor.
    *
    */
    virtual ~adjacency_list();


public:
    /**
    *
    * @brief   Returns amount of nodes in adjacency collection.
    *
    */
    virtual size_t size() const override;

    /**
    *
    * @brief   Establishes one-way connection from the first node to the second in adjacency collection.
    *
    * @details Complexity equals to complexity of insertion of std::unrodered_set. No bounds checking
    *          is performed.
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
    * @details Complexity equals to complexity of erasing of std::unrodered_set. No bounds checking
    *          is performed.
    *
    * @param[in]  node_index1: index of node in the collection that should be disconnected from another.
    * @param[in]  node_index2: index of another node in the collection that should be diconnected from
    *              the node defined by the first argument 'node_index1'.
    *
    */
    virtual void erase_connection(const size_t node_index1, const size_t node_index2) override;

    /**
    *
    * @brief   Checks existence of connection between specified nodes.
    *
    * @details Complexity equal to searching of std::unrodered_set. No bounds checking
    *          is performed.
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
    * @details Complexity equals to complexity of copying from unordered_set to vector. No bounds checking
    *          is performed.
    *
    * @param[in]  node_index: index of node in the collection whose neighbors are required.
    * @param[out] node_neighbors: vector of indexes of neighbors of specified node.
    *
    */
    virtual void get_neighbors(const size_t node_index, std::vector<size_t> & node_neighbors) const override;

    /**
    *
    * @brief   Clear content of adjacency matrix.
    *
    */
    virtual void clear() override;

public:
    adjacency_list & operator=(const adjacency_list & another_collection);

    adjacency_list & operator=(adjacency_list && another_collection);
};


}

}