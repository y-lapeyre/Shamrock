// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <climits>
#include <type_traits>

namespace shambase {

    template<typename T, int num>
    struct has_bitlen {
        static constexpr bool value = sizeof(T) * CHAR_BIT == num;
    };

    template<typename T, int num>
    inline constexpr bool has_bitlen_v = has_bitlen<T, num>::value;

} // namespace shambase