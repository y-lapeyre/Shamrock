// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"

TestStart(Unittest, "vtk-write", vtk_write_test, -1){

    shamrock::LegacyVtkWritter writter("out.vtk",true);

    

}