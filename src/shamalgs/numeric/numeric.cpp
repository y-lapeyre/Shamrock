// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "numeric.hpp"
#include "details/fallbackNumeric.hpp"
namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){
        return details::FallbackNumeric<T>::exclusive_sum(q, buf1, len);
    }

    template sycl::buffer<u32> exclusive_sum(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);


} // namespace shamalgs::numeric
