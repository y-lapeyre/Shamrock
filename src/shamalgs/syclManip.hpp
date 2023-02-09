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

namespace shamalgs::sycl_manip {



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

    template<class A,class B>
    struct VecConvert{
        inline static B convert(A arg){
            return arg;
        }
    };

    template<class A, class B>
    struct VecConvert<sycl::vec<A,2>, sycl::vec<B,2>>{
        inline static sycl::vec<B,2> convert(sycl::vec<A,2> arg){
            return {arg.x(),arg.y()};
        }
    };

    template<class A, class B>
    struct VecConvert<sycl::vec<A,3>, sycl::vec<B,3>>{
        inline static sycl::vec<B,3> convert(sycl::vec<A,3> arg){
            return {arg.x(),arg.y(),arg.z()};
        }
    };

    template<class A, class B>
    struct VecConvert<sycl::vec<A,4>, sycl::vec<B,4>>{
        inline static sycl::vec<B,4> convert(sycl::vec<A,4> arg){
            return {arg.x(),arg.y(),arg.z(),arg.w()};
        }
    };

    template<class A, class B>
    struct VecConvert<sycl::vec<A,8>, sycl::vec<B,8>>{
        inline static sycl::vec<B,8> convert(sycl::vec<A,8> arg){
            return {arg.s0(),arg.s1(),arg.s2(),arg.s3(),arg.s4(),arg.s5(),arg.s6(),arg.s7()};
        }
    };

    template<class A, class B>
    struct VecConvert<sycl::vec<A,16>, sycl::vec<B,16>>{
        inline static sycl::vec<B,16> convert(sycl::vec<A,16> arg){
            return {
                arg.s0(),arg.s1(),arg.s2(),arg.s3(),arg.s4(),arg.s5(),arg.s6(),arg.s7(),
                arg.s8(),arg.s9(),arg.sA(),arg.sB(),arg.sC(),arg.sD(),arg.sE(),arg.sF()
                };
        }
    };

    


}