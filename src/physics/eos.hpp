// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "patch/patchdata_buffer.hpp"
#include <memory>

enum EquationsOfStateType{
    isothermal
};

template<class flt>
class EquationsOfState {

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value || std::is_same<flt, f64>::value
    , "EquationsOfState : floating point type should be one of (f16,f32,f64)");

    inline static flt eos_isothermal(flt cs, flt rho){
        return cs*cs*rho;
    }



};