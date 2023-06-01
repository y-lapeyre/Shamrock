// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "SPHModel.hpp"

template<class Tvec, template<class> class SPHKernel>
f64 shammodels::SPHModel<Tvec,SPHKernel>::evolve_once(f64 dt_input, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id){
    return solver.evolve_once(dt_input, enable_physics, do_dump,vtk_dump_name,vtk_dump_patch_id);
}


using namespace shamrock::sph::kernels;

template class shammodels::SPHModel<f64_3,M4>;