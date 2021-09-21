/**
 * @file flags.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief The ONLY headers with conditional typedefs in the code
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 * 
 */

#pragma once

#include "aliases.hpp"


 #if defined(PRECISION_MORTON_DOUBLE)
    typedef u64 u_morton;
    typedef u32_3 u_ixyz;
    typedef i32_3 i_ixyz;
#else
    typedef u32 u_morton;
    typedef u16_3 u_ixyz;
    typedef i16_3 i_ixyz;
#endif

#if defined (PRECISION_FULL_SINGLE)
    typedef f32   f_s;
    typedef f32_2 f2_s;
    typedef f32_3 f3_s;
    typedef f32_4 f4_s;

    typedef f32   f_d;
    typedef f32_2 f2_d;
    typedef f32_3 f3_d;
    typedef f32_4 f4_d;
#endif

#if defined (PRECISION_MIXED)
    typedef f32   f_s;
    typedef f32_2 f2_s;
    typedef f32_3 f3_s;
    typedef f32_4 f4_s;

    typedef f64   f_d;
    typedef f64_2 f2_d;
    typedef f64_3 f3_d;
    typedef f64_4 f4_d;
#endif

#if defined (PRECISION_FULL_DOUBLE)
    typedef f64   f_s;
    typedef f64_2 f2_s;
    typedef f64_3 f3_s;
    typedef f64_4 f4_s;

    typedef f64   f_d;
    typedef f64_2 f2_d;
    typedef f64_3 f3_d;
    typedef f64_4 f4_d;
#endif
