#pragma once 

#include "../aliases.hpp"
#include "../flags.hpp"

void sort_morton_key_pair(
    sycl::queue* queue,
    u32 morton_count_rounded_pow,
    sycl::buffer<u32>*      buf_index,
    sycl::buffer<u_morton>* buf_morton
    );