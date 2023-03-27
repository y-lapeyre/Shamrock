// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"

#include <limits>

namespace shammath {

    template<class T>
    struct BBAA {

        using T_prop = shambase::sycl_utils::VectorProperties<T>;

        T lower;
        T upper;

        inline BBAA() = default;

        inline BBAA(T lower, T upper) : lower(lower), upper(upper){};

        inline BBAA(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline BBAA(std::pair<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline typename T_prop::component_type get_volume(){
            return shambase::product_accumulate(upper - lower);
        }

        inline BBAA get_intersect(BBAA other) const {
            return {
                shambase::sycl_utils::g_sycl_max(lower, other.lower),
                shambase::sycl_utils::g_sycl_min(upper, other.upper)
                };
        }

        inline BBAA is_empty(){
            return shambase::all_component_are_negative(upper - lower);
        }

        inline BBAA is_surface(){
            return shambase::component_have_only_one_zero(upper - lower);
        }

        inline BBAA is_surface_or_volume(){
            return shambase::component_have_at_most_one_zero(upper - lower);
        }
    };

}