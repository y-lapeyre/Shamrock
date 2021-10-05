#pragma once

#include "../../../aliases.hpp"
#include "../../../flags.hpp"

#include <vector>



void reduction_alg(
    //in
    sycl::queue* queue,
    u32 morton_count,
    sycl::buffer<u_morton>* buf_morton,
    u32 reduction_level,
    //out
    std::vector<u32> & reduc_index_map,
    u32 & morton_leaf_count);
    
void sycl_morton_remap_reduction(
    //in
    sycl::queue* queue,
    u32 morton_leaf_count,
    sycl::buffer<u32>* buf_reduc_index_map,
    sycl::buffer<u_morton>* buf_morton,
    //out
    sycl::buffer<u_morton>* buf_leaf_morton);