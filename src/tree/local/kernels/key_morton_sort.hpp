#pragma once 

#include "../../../aliases.hpp"
#include "../../../flags.hpp"

/**
 * @brief sort morton code and generate remap table 
 * 
 * @param queue sycl queue
 * @param morton_count_rounded_pow morton codes count (equal to a power of 2)
 * @param buf_index buffer countaining the index map corresponding to the sorting
 * @param buf_morton morton buffer that will be sorted
 */
void sycl_sort_morton_key_pair(
    sycl::queue & queue,
    u32 morton_count_rounded_pow,
    sycl::buffer<u32>*      buf_index,
    sycl::buffer<u_morton>* buf_morton
    );