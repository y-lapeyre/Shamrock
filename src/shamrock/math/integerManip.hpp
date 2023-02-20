// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamrock::math::int_manip {



    #ifdef SYCL_COMP_DPCPP

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{
        return sycl::clz(a);
    }

    #endif

    #ifdef SYCL_COMP_OPENSYCL

    namespace details{
        template<class T>
        int internal_clz(T a);

        template<>
        inline int internal_clz(u32 a){
            return __builtin_clz(a);
        }

        template<>
        inline int internal_clz(u64 a){
            return __builtin_clzl(a);
        }
    }

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{

        __hipsycl_if_target_host(
            return details::internal_clz(a);
        )

        __hipsycl_if_target_hiplike(
            return __clz(a);
        )

        __hipsycl_if_target_spirv(
            return __clz(a);
        )

    }

    #endif

    inline i32 clz_xor(u32 a, u32 b) noexcept {
        return (i32) sycl_clz(a^b);
    }

    template<class Acc>
    inline i32 karras_delta(i32 x, i32 y, u32 morton_lenght, Acc m) noexcept {
        return ((y > morton_lenght - 1 || y < 0) ? -1 : int(shamrock::math::int_manip::sycl_clz(m[x] ^ m[y])));
    }

    template<class T>
    constexpr T get_next_pow2_val (T val) noexcept;

    template<>
    inline constexpr u32 get_next_pow2_val (u32 val) noexcept {

        u32 clz_val = sycl_clz(val);

        u32 val_rounded_pow = 1U << (32-clz_val);
        if(val == 1U << (32-clz_val-1)){
            val_rounded_pow = val;
        }

        return val_rounded_pow; 
    };

    template<>
    inline constexpr u64 get_next_pow2_val (u64 val) noexcept {

        u64 clz_val = sycl_clz(val);

        u64 val_rounded_pow = 1U << (64-clz_val);
        if(val == 1U << (64-clz_val-1)){
            val_rounded_pow = val;
        }

        return val_rounded_pow; 
    };

}

