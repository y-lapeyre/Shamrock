// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_traits.hpp"
#include "vectorProperties.hpp"
#include "shambase/sycl.hpp"

namespace shambase::sycl_utils {

    template<class T>
    inline T g_sycl_min(T a, T b) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based) {
            return sycl::fmin(a, b);
        } else if constexpr (VectorProperties<T>::is_int_based) {
            return sycl::min(a, b);
        } else if constexpr (VectorProperties<T>::is_uint_based) {
            return sycl::min(a, b);
        }
    }

    template<class T>
    inline T g_sycl_max(T a, T b) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based) {
            return sycl::fmax(a, b);
        } else if constexpr (VectorProperties<T>::is_int_based) {
            return sycl::max(a, b);
        } else if constexpr (VectorProperties<T>::is_uint_based) {
            return sycl::max(a, b);
        }
    }



} // namespace shambase::sycl_utils