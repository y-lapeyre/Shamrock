// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "intervals.hpp"

#include <limits>

namespace shammath{

    template <class T> struct CoordRange {
        T lower;
        T upper;

        inline T delt() const { return upper - lower; }

        inline void expand_center(T tol){
            T center = (lower + upper) / 2;
            T cur_delt = upper - lower;
            cur_delt /= 2;
            cur_delt *= tol;
            lower = center - cur_delt;
            upper = center + cur_delt;
        }

        static CoordRange max_range();

    };

    template <> inline CoordRange<f32_3> CoordRange<f32_3>::max_range() {

        CoordRange<f32_3> ret;

        ret.lower = {
            std::numeric_limits<f32>::min(),
            std::numeric_limits<f32>::min(),
            std::numeric_limits<f32>::min()};

        ret.upper = {
            std::numeric_limits<f32>::max(),
            std::numeric_limits<f32>::max(),
            std::numeric_limits<f32>::max()};

        return ret;
    }

    template <> inline CoordRange<f64_3> CoordRange<f64_3>::max_range() {

        CoordRange<f64_3> ret;

        ret.lower = {
            std::numeric_limits<f64>::min(),
            std::numeric_limits<f64>::min(),
            std::numeric_limits<f64>::min()};

        ret.upper = {
            std::numeric_limits<f64>::max(),
            std::numeric_limits<f64>::max(),
            std::numeric_limits<f64>::max()};

        return ret;
    }


    template <> inline CoordRange<u32_3> CoordRange<u32_3>::max_range() {

        CoordRange<u32_3> ret;

        ret.lower = {
            std::numeric_limits<u32>::min(),
            std::numeric_limits<u32>::min(),
            std::numeric_limits<u32>::min()};

        ret.upper = {
            std::numeric_limits<u32>::max(),
            std::numeric_limits<u32>::max(),
            std::numeric_limits<u32>::max()};

        return ret;
    }

    template <> inline CoordRange<u64_3> CoordRange<u64_3>::max_range() {

        CoordRange<u64_3> ret;

        ret.lower = {
            std::numeric_limits<u64>::min(),
            std::numeric_limits<u64>::min(),
            std::numeric_limits<u64>::min()};

        ret.upper = {
            std::numeric_limits<u64>::max(),
            std::numeric_limits<u64>::max(),
            std::numeric_limits<u64>::max()};

        return ret;
    }

}