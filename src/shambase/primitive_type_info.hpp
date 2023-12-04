// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file memory.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/aliases_float.hpp"
#include <limits>

namespace shambase{

    template<class T>
    struct primitive_type_info{
        static constexpr bool is_specialized = false;

        static constexpr T max = {};
        static constexpr T min = {};
        static constexpr T epsilon = {};
        static constexpr T infty = {};
    };

    template<>
    struct primitive_type_info<f64>{
        using T = f64;

        static constexpr bool is_specialized = true;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest(); 
        static constexpr T epsilon = std::numeric_limits<T>::epsilon(); 
        static constexpr T infty = std::numeric_limits<T>::infinity();
    };



}