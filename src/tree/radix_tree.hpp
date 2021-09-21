#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include <vector>

class Radix_Tree{


    ///////////////////////////////////
    // This part is all about the reduction
    ///////////////////////////////////

    bool is_reduction_active = true;

    u16 reduction_level = 0;

    std::vector<u32> reduc_index_map;

    sycl::buffer<u_morton>* buf_reduced_morton  = nullptr;
    sycl::buffer<   u32  >* buf_reduc_index_map = nullptr;

    void do_reduction();


    ///////////////////////////////////
    // Normal radix tree
    ///////////////////////////////////

    u32 leaf_cell_count;
    sycl::buffer<u32>* buf_rchild_id   = nullptr;
    sycl::buffer<u32>* buf_lchild_id   = nullptr;
    sycl::buffer<u8 >* buf_rchild_flag = nullptr;
    sycl::buffer<u8 >* buf_lchild_flag = nullptr;
    sycl::buffer<u32>* buf_endrange    = nullptr;


};