// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"

namespace shambase {


    template<class T>
    inline constexpr T product_accumulate(T v) noexcept{
        return v;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T,n> v) noexcept{
        return v.x()*v.y();
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T,n> v) noexcept{
        return v.x()*v.y()*v.z();
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T,n> v) noexcept{
        return v.x()*v.y()*v.z()*v.w();
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T,n> v) noexcept{
        return v.s0()*v.s1()*v.s2()*v.s3()*v.s4()*v.s5()*v.s6()*v.s7();
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T,n> v) noexcept{
        return v.s0()*v.s1()*v.s2()*v.s3()*v.s4()*v.s5()*v.s6()*v.s7()*
        v.s8()*v.s9()*v.sA()*v.sB()*v.sC()*v.sD()*v.sE()*v.sF();
    }

}