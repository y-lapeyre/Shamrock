// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamutils::sycl_utils {

    template<class T>
    struct VectorProperties {
        using component_type           = T;
        static constexpr u32 dimension = 0;

        static constexpr bool is_float_based = std::is_same<T, f16>::value ||
                                               std::is_same<T, f32>::value ||
                                               std::is_same<T, f64>::value;
        static constexpr bool is_uint_based =
            std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based =
            std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
            std::is_same<T, i32>::value || std::is_same<T, i64>::value;

        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;
    };

    template<class T, u32 dim>
    struct VectorProperties<sycl::vec<T, dim>> {
        using component_type           = T;
        static constexpr u32 dimension = dim;

        static constexpr bool is_float_based = std::is_same<T, f16>::value ||
                                               std::is_same<T, f32>::value ||
                                               std::is_same<T, f64>::value;
        static constexpr bool is_uint_based =
            std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based =
            std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
            std::is_same<T, i32>::value || std::is_same<T, i64>::value;
        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;
    };

} // namespace shamutils::sycl_utils