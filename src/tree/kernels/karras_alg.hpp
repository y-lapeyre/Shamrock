#pragma once

#include "aliases.hpp"
#include "flags.hpp"
#include <memory>

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

template<class u_morton>
void sycl_karras_alg(
    sycl::queue & queue,
    u32 internal_cell_count,
    std::unique_ptr<sycl::buffer<u_morton>> & in_morton,
    std::unique_ptr<sycl::buffer<u32>> & out_buf_lchild_id   ,
    std::unique_ptr<sycl::buffer<u32>> & out_buf_rchild_id   ,
    std::unique_ptr<sycl::buffer<u8 >> & out_buf_lchild_flag ,
    std::unique_ptr<sycl::buffer<u8 >> & out_buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> & out_buf_endrange    );