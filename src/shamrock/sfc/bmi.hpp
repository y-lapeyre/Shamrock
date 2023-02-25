// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

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

#include "aliases.hpp"

namespace shamrock::sfc::bmi {

    template<class inttype,int interleaving>
    inttype expand_bits(inttype);



    //21bit number and 2 interleaving bits (for 64bits)
    template<>
    inline u64 expand_bits<u64,2>(u64 x){
        x &= 0x1fffffU;
        x = (x | x << 32U) & 0x1f00000000ffffU;
        x = (x | x << 16U) & 0x1f0000ff0000ffU;
        x = (x | x << 8U) & 0x100f00f00f00f00fU;
        x = (x | x << 4U) & 0x10c30c30c30c30c3U;
        x = (x | x << 2U) & 0x1249249249249249U;
        return x;
    }

    //10bit number and 2 interleaving bits (for 32 bits)
    template<>
    inline u32 expand_bits<u32,2>(u32 x){
        x &= 0x3ffU;
        x = (x | x << 16U) & 0x30000ffU;  
        x = (x | x << 8U) & 0x300f00fU;
        x = (x | x << 4U) & 0x30c30c3U;
        x = (x | x << 2U) & 0x9249249U;
        return x;
    }

    template<class inttype,int interleaving>
    inttype contract_bits(inttype);

    template<>
    inline u64 contract_bits<u64,2>(u64 src) {
        //src = src & 0x9249249249249249;
        src = (src | (src >> 2U))  & 0x30c30c30c30c30c3U;
        src = (src | (src >> 4U))  & 0xf00f00f00f00f00fU;
        src = (src | (src >> 8U))  & 0x00ff0000ff0000ffU;
        src = (src | (src >> 16U)) & 0xffff00000000ffffU;
        src = (src | (src >> 32U)) & 0x00000000ffffffffU;
        return src;
    }

    template<>
    inline u32 contract_bits<u32,2>(u32 src) {
        src = (src | src >> 2U) & 0x30C30C3U;
        src = (src | src >> 4U) & 0xF00F00FU;
        src = (src | src >> 8U) & 0xFF0000FFU;
        src = (src | src >> 16U) & 0xFFFFU;
        return src;
    }
    

} // namespace shamrock::sfc::bmi