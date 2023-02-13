// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include <random>

/**
 * @brief namespace to contain utility related to random number generation in shamalgs
 * 
 */
namespace shamalgs::random {

    template<class T> T mock_value(std::mt19937 & eng, T min_bound, T max_bound);
    template<class T> std::vector<T> mock_vector(u64 seed,u32 len, T min_bound, T max_bound);
    template<class T> sycl::buffer<T> mock_buffer(u64 seed,u32 len, T min_bound, T max_bound);

    template<class T>
    T next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval);

    template<>
    inline i64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i64(distval(eng));
    }
    template<>
    inline i32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i32(distval(eng));
    }
    template<>
    inline i16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i16(distval(eng));
    }
    template<>
    inline i8 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i8(distval(eng));
    }
    template<>
    inline u64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u64(distval(eng));
    }
    template<>
    inline u32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u32(distval(eng));
    }
    template<>
    inline u16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u16(distval(eng));
    }
    template<>
    inline u8 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u8(distval(eng));
    }
#ifdef SYCL_COMP_DPCPP
    template<>
    inline f16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f16(distval(eng));
    }
#endif
    template<>
    inline f32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f32(distval(eng));
    }
    template<>
    inline f64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f64(distval(eng));
    }

    template<>
    inline sycl::vec<f32, 2>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 2>{next_obj<f32>(eng, distval), next_obj<f32>(eng, distval)};
    }
    template<>
    inline sycl::vec<f32, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 3>{
            next_obj<f32>(eng, distval), next_obj<f32>(eng, distval), next_obj<f32>(eng, distval)};
    }
    template<>
    inline sycl::vec<f32, 4>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 4>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f32, 8>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 8>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f32, 16>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 16>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 2>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 2>{next_obj<f64>(eng, distval), next_obj<f64>(eng, distval)};
    }
    template<>
    inline sycl::vec<f64, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 3>{
            next_obj<f64>(eng, distval), next_obj<f64>(eng, distval), next_obj<f64>(eng, distval)};
    }
    template<>
    inline sycl::vec<f64, 4>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 4>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 8>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 8>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 16>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 16>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<u16, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<u16, 3>{
            next_obj<u16>(eng, distval), next_obj<u16>(eng, distval), next_obj<u16>(eng, distval)};
    }

    template<>
    inline sycl::vec<u32, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<u32, 3>{
            next_obj<u32>(eng, distval), next_obj<u32>(eng, distval), next_obj<u32>(eng, distval)};
    }

} // namespace shamalgs::random