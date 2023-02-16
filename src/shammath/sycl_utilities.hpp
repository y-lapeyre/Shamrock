#pragma once

#include "sycl_utils/vectorProperties.hpp"

namespace shammath::sycl_utils {

    template<class T>
    T g_sycl_min(T a, T b){

        static_assert(VectorProperties<T>::has_info, "no info about this type");
        
        if constexpr(VectorProperties<T>::is_float_based){
            return sycl::fmin(a,b);
        }else if constexpr(VectorProperties<T>::is_int_based){
            return sycl::min(a,b);
        }else if constexpr(VectorProperties<T>::is_uint_based){
            return sycl::min(a,b);
        }

    }

    template<class T>
    T g_sycl_max(T a, T b){

        static_assert(VectorProperties<T>::has_info, "no info about this type");
        
        if constexpr(VectorProperties<T>::is_float_based){
            return sycl::fmax(a,b);
        }else if constexpr(VectorProperties<T>::is_int_based){
            return sycl::max(a,b);
        }else if constexpr(VectorProperties<T>::is_uint_based){
            return sycl::max(a,b);
        }

    }
}