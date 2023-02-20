// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamutils/sycl_utilities.hpp"

namespace shamutils {

    /**
     * @brief return true if val is in [min,max[
     *
     * @tparam T
     * @param val
     * @param min
     * @param max
     * @return true
     * @return false
     */
    template<class T>
    inline bool is_in_half_open(T val, T min, T max) {
        return (val >= min) && (val < max);
    }

    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 2> val, sycl::vec<T, 2> min, sycl::vec<T, 2> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x()) && is_in_half_open(val.y(), min.y(), max.y())
        );
    }

    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 3> val, sycl::vec<T, 3> min, sycl::vec<T, 3> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x()) &&
            is_in_half_open(val.y(), min.y(), max.y()) && is_in_half_open(val.z(), min.z(), max.z())
        );
    }

    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 4> val, sycl::vec<T, 4> min, sycl::vec<T, 4> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x()) &&
            is_in_half_open(val.y(), min.y(), max.y()) &&
            is_in_half_open(val.z(), min.z(), max.z()) && is_in_half_open(val.w(), min.w(), max.w())
        );
    }

    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 8> val, sycl::vec<T, 8> min, sycl::vec<T, 8> max) {
        return (
            is_in_half_open(val.s0(), min.s0(), max.s0()) &&
            is_in_half_open(val.s1(), min.s1(), max.s1()) &&
            is_in_half_open(val.s2(), min.s2(), max.s2()) &&
            is_in_half_open(val.s3(), min.s3(), max.s3()) &&
            is_in_half_open(val.s4(), min.s4(), max.s4()) &&
            is_in_half_open(val.s5(), min.s5(), max.s5()) &&
            is_in_half_open(val.s6(), min.s6(), max.s6()) &&
            is_in_half_open(val.s7(), min.s7(), max.s7())
        );
    }

    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 16> val, sycl::vec<T, 16> min, sycl::vec<T, 16> max) {
        return (
            is_in_half_open(val.s0(), min.s0(), max.s0()) &&
            is_in_half_open(val.s1(), min.s1(), max.s1()) &&
            is_in_half_open(val.s2(), min.s2(), max.s2()) &&
            is_in_half_open(val.s3(), min.s3(), max.s3()) &&
            is_in_half_open(val.s4(), min.s4(), max.s4()) &&
            is_in_half_open(val.s5(), min.s5(), max.s5()) &&
            is_in_half_open(val.s6(), min.s6(), max.s6()) &&
            is_in_half_open(val.s7(), min.s7(), max.s7()) &&
            is_in_half_open(val.s8(), min.s8(), max.s8()) &&
            is_in_half_open(val.s9(), min.s9(), max.s9()) &&
            is_in_half_open(val.sA(), min.sA(), max.sA()) &&
            is_in_half_open(val.sB(), min.sB(), max.sB()) &&
            is_in_half_open(val.sC(), min.sC(), max.sC()) &&
            is_in_half_open(val.sD(), min.sD(), max.sD()) &&
            is_in_half_open(val.sE(), min.sE(), max.sE()) &&
            is_in_half_open(val.sF(), min.sF(), max.sF())
        );
    }

    template<class T>
    inline bool domain_are_connected(T bmin1, T bmax1, T bmin2, T bmax2);

    template<class T>
    inline bool domain_are_connected(
        sycl::vec<T, 3> bmin1, sycl::vec<T, 3> bmax1, sycl::vec<T, 3> bmin2, sycl::vec<T, 3> bmax2
    ) {
        return (
            (sycl_utils::g_sycl_max(bmin1.x(), bmin2.x()) <=
             sycl_utils::g_sycl_min(bmax1.x(), bmax2.x())) &&
            (sycl_utils::g_sycl_max(bmin1.y(), bmin2.y()) <=
             sycl_utils::g_sycl_min(bmax1.y(), bmax2.y())) &&
            (sycl_utils::g_sycl_max(bmin1.z(), bmin2.z()) <=
             sycl_utils::g_sycl_min(bmax1.z(), bmax2.z()))
        );
    }

} // namespace shamutils
