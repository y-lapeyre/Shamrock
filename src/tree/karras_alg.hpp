#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include "CL/sycl/buffer.hpp"

/**
 * @brief Karras 2012 algorithm with addition endrange buffer
 * 
 * @param internal_cell_count internal cell count
 * @param in_morton input morton set
 * @param buf_lchild_id output
 * @param buf_rchild_id output
 * @param buf_lchild_flag output
 * @param buf_rchild_flag output
 * @param buf_endrange output
 */

void karras_alg(
    sycl::queue* queue,
    u32 internal_cell_count,
    sycl::buffer<u_morton>* in_morton,
    sycl::buffer<u32>* out_buf_lchild_id   ,
    sycl::buffer<u32>* out_buf_rchild_id   ,
    sycl::buffer<u8 >* out_buf_lchild_flag ,
    sycl::buffer<u8 >* out_buf_rchild_flag,
    sycl::buffer<u32>* out_buf_endrange    );