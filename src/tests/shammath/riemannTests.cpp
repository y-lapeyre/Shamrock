// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shammath/riemann.hpp"
#include "shamtest/shamtest.hpp"
#include "shambase/aliases_float.hpp"

#include "shamcomm/logs.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"

TestStart(Unittest, "shammath/flux_symmetry", flux_rotate, 1){

    using Tcons = shammath::ConsState<f64_3>;

    Tcons state1 = {1._f64, 1.2_f64, f64_3{1,0,0}};
    Tcons state2 = {1.5_f64, 1._f64, f64_3{2,0,0}};

    {
        Tcons f1 = shammath::rusanov_flux_x(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_mx(state2, state1, 1.6666);
        _Assert(shambase::vec_equals(f1.rho,-f2.rho))
        _Assert(shambase::vec_equals(f1.rhovel,-f2.rhovel))
        _Assert(shambase::vec_equals(f1.rhoe,-f2.rhoe))
    }

    {
        Tcons f1 = shammath::rusanov_flux_y(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_my(state2, state1, 1.6666);
        _Assert(shambase::vec_equals(f1.rho,-f2.rho))
        _Assert(shambase::vec_equals(f1.rhovel,-f2.rhovel))
        _Assert(shambase::vec_equals(f1.rhoe,-f2.rhoe))
    }

    {
        Tcons f1 = shammath::rusanov_flux_z(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_mz(state2, state1, 1.6666);
        _Assert(shambase::vec_equals(f1.rho,-f2.rho))
        _Assert(shambase::vec_equals(f1.rhovel,-f2.rhovel))
        _Assert(shambase::vec_equals(f1.rhoe,-f2.rhoe))
    }

}