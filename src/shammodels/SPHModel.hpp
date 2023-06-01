// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/SPHModelSolver.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels {

    /**
     * @brief The shamrock SPH model
     * 
     * @tparam Tvec 
     * @tparam SPHKernel 
     */
    template<class Tvec, template<class> class SPHKernel>
    class SPHModel {public:

        using Tscal                    = shambase::VecComponent<Tvec>;
        static constexpr u32 dimension = shambase::VectorProperties<Tvec>::dimension;
        using Kernel = SPHKernel<Tscal>;

        using Solver = SPHModelSolver<Tvec, SPHKernel>;
        //using SolverConfig = typename Solver::Config;

        ShamrockCtx & ctx;

        Solver solver;

        //SolverConfig sconfig;

        SPHModel(ShamrockCtx & ctx) : ctx(ctx), solver(ctx) {};

        /////// setup function

        inline void init_scheduler(u32 crit_split, u32 crit_merge){
            solver.init_required_fields();
            ctx.init_sched(crit_split, crit_merge);
        }


        //inline void enable_barotropic_mode(){
        //    sconfig.enable_barotropic();
        //}
        //
        //inline void switch_internal_energy_mode(std::string name){
        //    sconfig.switch_internal_energy_mode(name);
        //}

        

        /////// analysis utilities

        /////// I/O

        /////// Simulation control

        f64 evolve_once(f64 dt_input, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id);
    };

} // namespace shammodels