// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"

namespace shambase {



    template<class T,class AccU8>
    inline void store_u8(AccU8 & acc, u64 ptr_write, T a){
        constexpr u64 szT = sizeof(T);
        u8 *bytes = (u8 *)&a;
        #pragma unroll
        for(u64 i = 0; i < szT ; i++){
            acc[ptr_write+i] = bytes[i];
        }
    }

    template<class T,class AccU8>
    inline T load_u8(AccU8 & acc, u64 ptr_load){
        constexpr u64 szT = sizeof(T);
        T ret;
        u8 *bytes = (u8 *)&ret;
        #pragma unroll
        for(u64 i = 0; i < szT ; i++){
            bytes[i] = acc[ptr_load + i];
        }
        return ret;
    }

    template<class T, class TAcc>
    inline void store_conv(TAcc* acc, T a){
        T* ptr = (T*) acc;
        *ptr = a;
    }

    template<class T,class TAcc>
    inline T load_conv(TAcc* acc){
        T* ptr = (T*) acc;
        return *ptr;
    }

}