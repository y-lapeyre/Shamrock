// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/integer.hpp"
#include "shambase/type_aliases.hpp"

namespace shambase {

    // p. 45 of Pointers in C:
    inline bool little_endian() {
        short int word = 0x0001;
        char *byte     = (char *)&word;
        return (byte[0] ? 1 : 0);
    }

    template<class T>
    inline void endian_swap(T &a) {

        constexpr i32 sz = sizeof(a);

        auto constexpr lambd = []() {
            if constexpr (sz % 2 == 0) {
                return sz / 2;
            } else {
                return (sz - 1) / 2;
            }
        };

        constexpr i32 steps = lambd();

        u8 *bytes = (u8 *)&a;

        for (i32 i = 0; i < steps; i++) {
            xor_swap(bytes[i], bytes[sz - 1 - i]);
        }
    }
    

} // namespace shambase