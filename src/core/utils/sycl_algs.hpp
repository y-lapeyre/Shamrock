#pragma once

#include "CL/sycl/buffer.hpp"
#include "aliases.hpp"
#include "core/sys/sycl_handler.hpp"

namespace syclalgs {

    namespace basic {

        template <class T> void copybuf(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template <class T> void copybuf_discard(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template<class T>
        void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count);

    } // namespace basic

    namespace reduction {

        bool is_all_true(sycl::buffer<u8> & buf);
        
    }

} // namespace syclalgs