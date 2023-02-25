// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "reduction.hpp"

#include "shamrock/legacy/algs/sycl/basic/basic.hpp"
#include "shamalgs/reduction/details/sycl2020reduction.hpp"
#include "shamalgs/reduction/details/groupReduction.hpp"
#include "shamalgs/reduction/details/fallbackReduction.hpp"

namespace shamalgs::reduction {


    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::SYCL2020<T>::sum(q, buf1, start_id, end_id);
    }

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::FallbackReduction<T>::max(q, buf1, start_id, end_id);
    }

    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::FallbackReduction<T>::min(q, buf1, start_id, end_id);
    }


    #define XMAC_TYPES \
    X(f32   ) \
    X(f32_2 ) \
    X(f32_3 ) \
    X(f32_4 ) \
    X(f32_8 ) \
    X(f32_16) \
    X(f64   ) \
    X(f64_2 ) \
    X(f64_3 ) \
    X(f64_4 ) \
    X(f64_8 ) \
    X(f64_16) \
    X(u32   ) \
    X(u64   ) \
    X(u32_3 ) \
    X(u64_3 )

    #define X(_arg_) \
    template _arg_ sum(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);\
    template _arg_ max(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);\
    template _arg_ min(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);

    XMAC_TYPES
    #undef X

} // namespace shamalgs::reduction