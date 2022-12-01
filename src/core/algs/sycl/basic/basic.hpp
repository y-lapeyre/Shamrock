#pragma once

#include "aliases.hpp"
#include "core/sys/sycl_handler.hpp"

namespace syclalgs::basic {

    template <class T> void copybuf(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

    template <class T> void copybuf_discard(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

    template<class T>
    void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count);

} // namespace syclalgs::basic