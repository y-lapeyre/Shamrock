// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sphkernels.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include <builtins.hpp>
namespace shamphys {

    template<class T>
    T hill_radius(T R, T m, T M) {
        return R * sycl::cbrt(m / (3 * M));
    }

    template<class T>
    T keplerian_speed(T G, T M, T R){
        return sycl::sqrt(G * M / R);
    }

} // namespace shamphys