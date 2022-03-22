#pragma once 

#include "aliases.hpp"
#include "flags.hpp"
#include <memory>

/**
 * @brief sort morton code and generate remap table 
 *
 * @tparam u_morton morton precision
 * @param queue sycl queue
 * @param morton_count_rounded_pow morton codes count (equal to a power of 2)
 * @param buf_index buffer countaining the index map corresponding to the sorting
 * @param buf_morton morton buffer that will be sorted
 */
template<class u_morton>
void sycl_sort_morton_key_pair(
    sycl::queue & queue,
    u32 morton_count_rounded_pow,
    std::unique_ptr<sycl::buffer<u32>>      & buf_index,
    std::unique_ptr<sycl::buffer<u_morton>> & buf_morton
    );