// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

inline constexpr f64 hfact = 1.2;

template<class flt>
inline flt rho_h(flt m, flt h){
    constexpr flt hfac = hfact;
    return m*(hfac/h)*(hfac/h)*(hfac/h);
}

template<class flt>
inline flt h_rho(flt m, flt rho){
    constexpr flt hfac = hfact;
    return hfac/sycl::pow(rho/m, 1/3);
}