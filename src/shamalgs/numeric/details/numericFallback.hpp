// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    sycl::buffer<T> exclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);
    
    template<class T>
    sycl::buffer<T> inclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    std::tuple<sycl::buffer<u32>, u32>
    stream_compact_fallback(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric::details
