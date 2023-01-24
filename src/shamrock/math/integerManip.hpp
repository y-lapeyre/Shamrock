#pragma once

#include "aliases.hpp"

namespace shamrock::math::int_manip {



    #ifdef SYCL_COMP_DPCPP

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{
        return sycl::clz(a);
    }

    #endif

    #ifdef SYCL_COMP_HIPSYCL

    namespace details{
        template<class T>
        int internal_clz(T a);

        template<>
        int internal_clz(u32 a){
            return __builtin_clz(a);
        }

        template<>
        int internal_clz(u64 a){
            return __builtin_clzl(a);
        }
    }

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{
        return 
            __hipsycl_if_target_host(details::internal_clz(a))
            __hipsycl_if_target_cuda(__clz(a))
            __hipsycl_if_target_hip(__clz(a))
            __hipsycl_if_target_spirv(__clz(a))
            ;
    }

    #endif




    inline constexpr u32 get_next_pow2_val (u32 val) noexcept {

        u32 clz_val = sycl_clz(val);

        u32 val_rounded_pow = 1U << (32-clz_val);
        if(val == 1U << (32-clz_val-1)){
            val_rounded_pow = val;
        }

        return val_rounded_pow; 
    };

}

