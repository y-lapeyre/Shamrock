// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamtest/shamtest.hpp"

#include "shambase/Constants.hpp"

template class shambase::Constants<f32>;
template class shambase::Constants<f64>;

TestStart(Unittest, "shambase/Constants", checkconstantmatchsycl, 1){
    _AssertFloatEqual(shambase::Constants<f32>::pi, 4*sycl::atan(f32(1)), 1e-25);
    _AssertFloatEqual(shambase::Constants<f64>::pi, 4*sycl::atan(f64(1)), 1e-25);
}