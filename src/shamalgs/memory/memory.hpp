// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamalgs::memory {


    /**
     * @brief extract a value of a buffer 
     * 
     * @tparam T the type of the buffer & value
     * @param q the queue to use
     * @param buf the buffer to extract from
     * @param idx the index of the value that will be extracted
     * @return T the extracted value
     */
    template<class T> T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

    template<class T> sycl::buffer<T> vec_to_buf(const std::vector<T> &buf);
    template<class T> std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);


    /**
     * @brief enqueue a do nothing kernel to force the buffer to move
     * 
     * @tparam T 
     * @param q 
     * @param buf 
     */
    template<class T> inline void move_buffer_on_queue(sycl::queue & q, sycl::buffer<T> & buf){
        q.submit([&](sycl::handler & cgh){
            cgh.require(sycl::accessor{buf, cgh, sycl::read_write});
        });
    }
}