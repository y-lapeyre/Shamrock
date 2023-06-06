// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "SPHModelSolver.hpp"

template<class Tvec, template<class> class Kern>
using SPHSolve = shammodels::SPHModelSolver<Tvec, Kern>;

template<class Tvec, template<class> class Kern>
auto SPHSolve<Tvec, Kern>::evolve_once(Tscal dt_input,
                                       bool enable_physics,
                                       bool do_dump,
                                       std::string vtk_dump_name,
                                       bool vtk_dump_patch_id) -> Tscal {
    tmp_solver.set_cfl_cour(cfl_cour);
    tmp_solver.set_cfl_force(cfl_force);
    tmp_solver.set_particle_mass(gpart_mass);
    // tmp_solver.set_gamma(eos_gamma);

    struct DumpOption{
        bool vtk_do_dump;
        std::string vtk_dump_fname;
        bool vtk_dump_patch_id;
    };




    return tmp_solver.evolve(dt_input, enable_physics, {do_dump, vtk_dump_name, vtk_dump_patch_id});
}

using namespace shamrock::sph::kernels;

template class shammodels::SPHModelSolver<f64_3, M4>;