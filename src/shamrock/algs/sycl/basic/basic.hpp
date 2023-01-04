// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamsys/sycl_handler.hpp"

namespace syclalgs::basic {

    template <class T> void copybuf(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

    template <class T> void copybuf_discard(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

    template<class T>
    void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count);

    template<class T> std::unique_ptr<sycl::buffer<T>> duplicate(const std::unique_ptr<sycl::buffer<T>> & vec);

} // namespace syclalgs::basic