// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamsys/Log.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamsys/Log", test_format, 1){


    std::cout << shamsys::format("{} 1", f64_3{0,1,2}) << std::endl;

}