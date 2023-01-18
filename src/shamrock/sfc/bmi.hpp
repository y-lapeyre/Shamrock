// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file bmi.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Bit manipulation instruction implementation for SYCL
 * @version 0.1
 * @date 2022-03-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include "aliases.hpp"

namespace shamrock::sfc::bmi {

    template<class inttype,int interleaving>
    inttype expand_bits(inttype);



    //21bit number and 2 interleaving bits (for 64bits)
    template<>
    inline u64 expand_bits<u64,2>(u64 x){
        x &= 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }

    //10bit number and 2 interleaving bits (for 32 bits)
    template<>
    inline u32 expand_bits<u32,2>(u32 x){
        x &= 0x3ff;
        x = (x | x << 16) & 0x30000ff;  
        x = (x | x << 8) & 0x300f00f;
        x = (x | x << 4) & 0x30c30c3;
        x = (x | x << 2) & 0x9249249;
        return x;
    }

    template<class inttype,int interleaving>
    inttype contract_bits(inttype);

    template<>
    inline u64 contract_bits<u64,2>(u64 src) {
        //src = src & 0x9249249249249249;
        src = (src | (src >> 2))  & 0x30c30c30c30c30c3;
        src = (src | (src >> 4))  & 0xf00f00f00f00f00f;
        src = (src | (src >> 8))  & 0x00ff0000ff0000ff;
        src = (src | (src >> 16)) & 0xffff00000000ffff;
        src = (src | (src >> 32)) & 0x00000000ffffffff;
        return src;
    }

    template<>
    inline u32 contract_bits<u32,2>(u32 src) {
        src = (src | src >> 2) & 0x30C30C3;
        src = (src | src >> 4) & 0xF00F00F;
        src = (src | src >> 8) & 0xFF0000FF;
        src = (src | src >> 16) & 0xFFFF;
        return src;
    }

} // namespace shamrock::sfc::bmi