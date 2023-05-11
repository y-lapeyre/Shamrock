// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

//%Impl status : Good

#include "shambase/sycl.hpp"
namespace shamrock::sph {

    constexpr f64 default_hfact = 1.2;

    template<class flt>
    inline flt rho_h(flt m, flt h, flt hfact = default_hfact) {
        return m * (hfact / h) * (hfact / h) * (hfact / h);
    }

    template<class flt>
    inline flt h_rho(flt m, flt rho, flt hfact = default_hfact) {
        return hfact / sycl::rootn(rho / m, 3);
    }
} // namespace shamrock::sph