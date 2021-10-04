#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include "CL/sycl/buffer.hpp"
#include <vector>

//TODO write class destructor

class Radix_Tree{



    ///////////////////////////////////
    // Normal radix tree
    ///////////////////////////////////

    /**
     * @brief true if leaf_cell_count == 1
     */
    bool mono_cell_mode = false;

    sycl::buffer<u32>* sort_index_map = nullptr;


    u32 leaf_cell_count = 0;
    u32 internal_cell_count = 0;

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


    /**
     * @brief build tree (warning buf_morton will be sorted)
     * 
     * @param queue sycl queue
     * @param buf_morton morton code in buffer (warning : it will be sorted) 
     * @param morton_code_count number of valid morton codes
     * @param morton_code_count_rounded_pow lenght of buf_morton
     * @param use_reduction use reduction algorithm
     * @param reduction_level level of reduction
     */
    void build_tree(
        sycl::queue* queue,
        sycl::buffer<u_morton>* buf_morton, 
        u32 morton_code_count, 
        u32 morton_code_count_rounded_pow,
        bool use_reduction, 
        u32 reduction_level);









    ///////////////////////////////////
    // This part is all about the reduction
    ///////////////////////////////////

    bool is_reduction_active = true;
    u16 reduction_level = 0;

    float reduction_factor = 0;

    std::vector<u32> reduc_index_map;
    sycl::buffer<   u32  >* buf_reduc_index_map = nullptr;



};