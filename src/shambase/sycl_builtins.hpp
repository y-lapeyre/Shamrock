// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"

namespace shambase {

    /**
     * @brief Fallback function for sycl::any
     *
     * @tparam T
     * @tparam n
     * @param v
     * @return i32
     */
    template<
        class T,
        int n,
        std::enable_if_t<std::is_integral_v<T>,
                         int> = 0>
    i32 any(sycl::vec<T, n> v) {
        #ifdef SYCL_COMP_DPCPP
            return sycl::any(v);
        #else
            return shambase::sum_accumulate(sycl::abs(v)) > 0;
        #endif
    }

} // namespace shambase