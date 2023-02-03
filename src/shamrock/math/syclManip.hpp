// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "vectorManip.hpp"

namespace shamrock::math::sycl_manip {



    template<class T>
    T g_sycl_min(T a, T b){

        static_assert(vec_manip::VectorProperties<T>::has_info, "no info about this type");
        
        if constexpr(vec_manip::VectorProperties<T>::is_float_based){
            return sycl::fmin(a,b);
        }else if constexpr(vec_manip::VectorProperties<T>::is_int_based){
            return sycl::min(a,b);
        }else if constexpr(vec_manip::VectorProperties<T>::is_uint_based){
            return sycl::min(a,b);
        }

    }

    template<class T>
    T g_sycl_max(T a, T b){

        static_assert(vec_manip::VectorProperties<T>::has_info, "no info about this type");
        
        if constexpr(vec_manip::VectorProperties<T>::is_float_based){
            return sycl::fmax(a,b);
        }else if constexpr(vec_manip::VectorProperties<T>::is_int_based){
            return sycl::max(a,b);
        }else if constexpr(vec_manip::VectorProperties<T>::is_uint_based){
            return sycl::max(a,b);
        }

    }


}