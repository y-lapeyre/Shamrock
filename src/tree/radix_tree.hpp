#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include <vector>

class Radix_Tree{



    ///////////////////////////////////
    // Normal radix tree
    ///////////////////////////////////

    /**
     * @brief true if leaf_cell_count == 1
     */
    bool mono_cell_mode = false;

    u32 leaf_cell_count;
    u32 internal_cell_count;

    /**
     * @brief list of morton codes of the tree leafs (so reduced morton set if reduction applied)
     */
    sycl::buffer<u_morton>* buf_leaf_morton   = nullptr;
    
    sycl::buffer<u32>* buf_rchild_id   = nullptr;
    sycl::buffer<u32>* buf_lchild_id   = nullptr;
    sycl::buffer<u8 >* buf_rchild_flag = nullptr;
    sycl::buffer<u8 >* buf_lchild_flag = nullptr;
    sycl::buffer<u32>* buf_endrange    = nullptr;

    sycl::buffer<u_ixyz>* buf_ipos_min = nullptr;
    sycl::buffer<u_ixyz>* buf_ipos_max = nullptr;

    sycl::buffer<f3_d>* buf_pos_min    = nullptr;
    sycl::buffer<f3_d>* buf_pos_max    = nullptr;


    void build_tree();









    ///////////////////////////////////
    // This part is all about the reduction
    ///////////////////////////////////

    bool is_reduction_active = true;
    u16 reduction_level = 0;

    sycl::buffer<   u32  >* buf_reduc_index_map = nullptr;



};