// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sycl_vector_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once


#include "aliases.hpp"
template <class T> inline bool test_eq1(T a, T b) {
    bool eqx = a == b;
    return eqx;
}

template <class T> inline bool test_eq2(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    return eqx && eqy;
}

template <class T> inline bool test_eq3(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    return eqx && eqy && eqz;
}

template <class T> inline bool test_eq4(T a, T b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    bool eqw = a.w() == b.w();
    return eqx && eqy && eqz && eqw;
}

template <class T> inline bool test_eq8(T a, T b) {
    bool eqs0 = a.s0() == b.s0();
    bool eqs1 = a.s1() == b.s1();
    bool eqs2 = a.s2() == b.s2();
    bool eqs3 = a.s3() == b.s3();
    bool eqs4 = a.s4() == b.s4();
    bool eqs5 = a.s5() == b.s5();
    bool eqs6 = a.s6() == b.s6();
    bool eqs7 = a.s7() == b.s7();
    return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7;
}

template <class T> inline bool test_eq16(T a, T b) {
    bool eqs0 = a.s0() == b.s0();
    bool eqs1 = a.s1() == b.s1();
    bool eqs2 = a.s2() == b.s2();
    bool eqs3 = a.s3() == b.s3();
    bool eqs4 = a.s4() == b.s4();
    bool eqs5 = a.s5() == b.s5();
    bool eqs6 = a.s6() == b.s6();
    bool eqs7 = a.s7() == b.s7();

    bool eqs8 = a.s8() == b.s8();
    bool eqs9 = a.s9() == b.s9();
    bool eqsA = a.sA() == b.sA();
    bool eqsB = a.sB() == b.sB();
    bool eqsC = a.sC() == b.sC();
    bool eqsD = a.sD() == b.sD();
    bool eqsE = a.sE() == b.sE();
    bool eqsF = a.sF() == b.sF();

    return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7 && eqs8 && eqs9 && eqsA && eqsB && eqsC && eqsD &&
           eqsE && eqsF;
}



template <class T> inline bool test_sycl_eq(T a, T b);


template <> inline bool test_sycl_eq<f32>(f32 a, f32 b){
    return test_eq1(a, b);
}

template <> inline bool test_sycl_eq<f32_2>(f32_2 a, f32_2 b){
    return test_eq2(a, b);
}

template <> inline bool test_sycl_eq<f32_3>(f32_3 a, f32_3 b){
    return test_eq3(a, b);
}

template <> inline bool test_sycl_eq<f32_4>(f32_4 a, f32_4 b){
    return test_eq4(a, b);
}

template <> inline bool test_sycl_eq<f32_8>(f32_8 a, f32_8 b){
    return test_eq8(a, b);
}

template <> inline bool test_sycl_eq<f32_16>(f32_16 a, f32_16 b){
    return test_eq16(a, b);
}


template <> inline bool test_sycl_eq<f64>(f64 a, f64 b){
    return test_eq1(a, b);
}

template <> inline bool test_sycl_eq<f64_2>(f64_2 a, f64_2 b){
    return test_eq2(a, b);
}

template <> inline bool test_sycl_eq<f64_3>(f64_3 a, f64_3 b){
    return test_eq3(a, b);
}

template <> inline bool test_sycl_eq<f64_4>(f64_4 a, f64_4 b){
    return test_eq4(a, b);
}

template <> inline bool test_sycl_eq<f64_8>(f64_8 a, f64_8 b){
    return test_eq8(a, b);
}

template <> inline bool test_sycl_eq<f64_16>(f64_16 a, f64_16 b){
    return test_eq16(a, b);
}

template <> inline bool test_sycl_eq<u32>(u32 a, u32 b){
    return test_eq1(a, b);
}

template <> inline bool test_sycl_eq<u32_2>(u32_2 a, u32_2 b){
    return test_eq2(a, b);
}

template <> inline bool test_sycl_eq<u32_3>(u32_3 a, u32_3 b){
    return test_eq3(a, b);
}

template <> inline bool test_sycl_eq<u32_4>(u32_4 a, u32_4 b){
    return test_eq4(a, b);
}

template <> inline bool test_sycl_eq<u32_8>(u32_8 a, u32_8 b){
    return test_eq8(a, b);
}

template <> inline bool test_sycl_eq<u32_16>(u32_16 a, u32_16 b){
    return test_eq16(a, b);
}

template <> inline bool test_sycl_eq<u64>(u64 a, u64 b){
    return test_eq1(a, b);
}

template <> inline bool test_sycl_eq<u64_2>(u64_2 a, u64_2 b){
    return test_eq2(a, b);
}

template <> inline bool test_sycl_eq<u64_3>(u64_3 a, u64_3 b){
    return test_eq3(a, b);
}

template <> inline bool test_sycl_eq<u64_4>(u64_4 a, u64_4 b){
    return test_eq4(a, b);
}

template <> inline bool test_sycl_eq<u64_8>(u64_8 a, u64_8 b){
    return test_eq8(a, b);
}

template <> inline bool test_sycl_eq<u64_16>(u64_16 a, u64_16 b){
    return test_eq16(a, b);
}