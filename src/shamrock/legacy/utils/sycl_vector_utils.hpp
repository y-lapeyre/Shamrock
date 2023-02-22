// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

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



#include "aliases.hpp"
#include <ostream>
#include <random> 


template <class T> inline void print_vec(std::ostream & ostream, T a);

template<class T> inline T next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval);






template <class T> inline bool test_sycl_eq(const T & a, const T & b);

template <class T> inline bool test_sycl_eq(const sycl::vec<T, 2> & a, const sycl::vec<T, 2> & b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    return eqx && eqy;
}

template <class T> inline bool test_sycl_eq(const sycl::vec<T, 3> & a, const sycl::vec<T, 3> & b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    return eqx && eqy && eqz;
}

template <class T> inline bool test_sycl_eq(const sycl::vec<T, 4> & a, const sycl::vec<T, 4> & b) {
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    bool eqw = a.w() == b.w();
    return eqx && eqy && eqz && eqw;
}

template <class T> inline bool test_sycl_eq(const sycl::vec<T, 8> & a, const sycl::vec<T, 8> & b) {
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

template <class T> inline bool test_sycl_eq(const sycl::vec<T, 16> & a, const sycl::vec<T, 16> & b) {
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



template <class T> inline bool test_sycl_eq(const T & a, const T & b) {
    bool eqx = a == b;
    return eqx;
}



































template<class T> inline void print_vec(std::ostream &ostream, sycl::vec<T,2> a){
    ostream << "("<< a.x() <<","<< a.y()<<")";
}

template<class T> inline void print_vec(std::ostream &ostream, sycl::vec<T,3> a){
    ostream << "("<< a.x() <<","<< a.y()<<","<< a.z()<<")";
}

template<class T> inline void print_vec(std::ostream &ostream, sycl::vec<T,4> a){
    ostream << "("<< a.x() <<","<< a.y()<<","<< a.z()<<","<< a.w()<<")";
}

template<class T> inline void print_vec(std::ostream &ostream, sycl::vec<T,8> a){
    ostream << "("<< a.s0() <<","<< a.s1()<<","<< a.s2()<<","<< a.s3()<<
    a.s4() <<","<< a.s5()<<","<< a.s6()<<","<< a.s7()<<")";
}

template<class T> inline void print_vec(std::ostream &ostream, sycl::vec<T,16> a){
    ostream << "("<< a.s0() <<","<< a.s1()<<","<< a.s2()<<","<< a.s3()<<
    a.s4() <<","<< a.s5()<<","<< a.s6()<<","<< a.s7()<<
    a.s8() <<","<< a.s9()<<","<< a.sA()<<","<< a.sB()<<
    a.sC() <<","<< a.sD()<<","<< a.sE()<<","<< a.sF()<<")";
}


template<class T> inline void print_vec(std::ostream &ostream, T a){
    ostream << a;
}









template<> inline i64 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return i64(distval(eng));}
template<> inline i32 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return i32(distval(eng));}
template<> inline i16 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return i16(distval(eng));}
template<> inline i8  next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return i8 (distval(eng));}
template<> inline u64 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return u64(distval(eng));}
template<> inline u32 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return u32(distval(eng));}
template<> inline u16 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return u16(distval(eng));}
template<> inline u8  next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return u8 (distval(eng));}
#ifdef SYCL_COMP_DPCPP
template<> inline f16 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return f16(distval(eng));}
#endif
template<> inline f32 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return f32(distval(eng));}
template<> inline f64 next_obj(std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval){ return f64(distval(eng));}




template<> inline sycl::vec<f32,2> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f32,2> {next_obj<f32>(eng,distval),next_obj<f32>(eng,distval)};}
template<> inline sycl::vec<f32,3> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f32,3> {next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval)};}
template<> inline sycl::vec<f32,4> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f32,4> {next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval)};}

template<> inline sycl::vec<f32,8> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){
    return sycl::vec<f32,8> {
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval)
    };
}

template<> inline sycl::vec<f32,16> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){
    return sycl::vec<f32,16> {
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),
        next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval),next_obj<f32>(eng,distval)
    };
}




template<> inline sycl::vec<f64,2> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f64,2> {next_obj<f64>(eng,distval),next_obj<f64>(eng,distval)};}
template<> inline sycl::vec<f64,3> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f64,3> {next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval)};}
template<> inline sycl::vec<f64,4> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<f64,4> {next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval)};}

template<> inline sycl::vec<f64,8> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){
    return sycl::vec<f64,8> {
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval)
    };
}

template<> inline sycl::vec<f64,16> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){
    return sycl::vec<f64,16> {
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),
        next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval),next_obj<f64>(eng,distval)
    };
}






template<> inline sycl::vec<u16,3> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<u16,3> {next_obj<u16>(eng,distval),next_obj<u16>(eng,distval),next_obj<u16>(eng,distval)};}


template<> inline sycl::vec<u32,3> next_obj(std::mt19937 & eng, std::uniform_real_distribution<f64> & distval){return sycl::vec<u32,3> {next_obj<u32>(eng,distval),next_obj<u32>(eng,distval),next_obj<u32>(eng,distval)};}
