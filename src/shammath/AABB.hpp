// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AABB.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/SourceLocation.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"
#include "shambackends/math.hpp"

#include <limits>

namespace shammath {

    /**
     * @brief Axis-Aligned bounding box
     * This class describe a bounding box aligned on the axis
     * This class describe a domain of coordinates defined by a cartesian product of 1d ranges
     * For exemple : [ax,bx] x [ay,by] x [az,bz]
     * 
     * @tparam T 
     */
    template<class T>
    struct AABB {

        using T_prop = shambase::VectorProperties<T>;

        T lower;
        T upper;

        inline AABB() = default;

        inline AABB(T lower, T upper) : lower(lower), upper(upper){};

        inline AABB(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline AABB(std::pair<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline typename T_prop::component_type get_volume(){
            return shambase::product_accumulate(upper - lower);
        }

        inline T get_center() const noexcept{
            return (lower + upper)/2;
        }

        inline T sum_bounds() const noexcept{
            return lower + upper;
        }

        inline AABB get_intersect(AABB other)  const noexcept {
            return {
                sham::max(lower, other.lower),
                sham::min(upper, other.upper)
                };
        }

        [[nodiscard]] inline bool is_not_empty() const noexcept{
            return shambase::vec_compare_geq(upper , lower);
        }

        [[nodiscard]] inline bool is_volume_not_null() const noexcept{
            return shambase::vec_compare_g(upper , lower);
        }

        [[nodiscard]] inline bool is_surface() const noexcept{
            return shambase::component_have_only_one_zero(delt()) && (is_not_empty());
        }

        [[nodiscard]] inline bool is_surface_or_volume() const noexcept{
            return shambase::component_have_at_most_one_zero(delt()) && (is_not_empty());
        }
    };

}