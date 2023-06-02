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
#include "shammodels/generic/setup/generators.hpp"
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
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel = SPHKernel<Tscal>;

        using Solver = SPHModelSolver<Tvec, SPHKernel>;
        //using SolverConfig = typename Solver::Config;

        ShamrockCtx & ctx;

        Solver solver;

        //SolverConfig sconfig;

        SPHModel(ShamrockCtx & ctx) : ctx(ctx), solver(ctx) {};


        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// setup function
        ////////////////////////////////////////////////////////////////////////////////////////////

        inline void init_scheduler(u32 crit_split, u32 crit_merge){
            solver.init_required_fields();
            ctx.init_sched(crit_split, crit_merge);
        }




        template<std::enable_if_t<dim == 3, int> = 0>
        inline Tvec get_box_dim_fcc_3d(Tscal dr, u32 xcnt, u32 ycnt, u32 zcnt){
            return generic::setup::generators::get_box_dim(dr, xcnt, ycnt, zcnt);
        }

        inline void set_cfl_cour(Tscal cfl_cour){
            solver.tmp_solver.set_cfl_cour(cfl_cour);
        }
        inline void set_cfl_force(Tscal cfl_force){
            solver.tmp_solver.set_cfl_force(cfl_force);
        }
        inline void set_particle_mass(Tscal gpart_mass){
            solver.tmp_solver.set_particle_mass(gpart_mass);
        }

        template<std::enable_if_t<dim == 3, int> = 0>
        inline std::pair<Tvec,Tvec> get_ideal_fcc_box(Tscal dr, std::pair<Tvec,Tvec> box){
            auto [a,b] =  generic::setup::generators::get_ideal_fcc_box<Tscal>(dr, box);
            return {a,b};
        }

        inline void resize_simulation_box(std::pair<Tvec,Tvec> box){
            ctx.set_coord_domain_bound({box.first, box.second});
        }


        //inline void enable_barotropic_mode(){
        //    sconfig.enable_barotropic();
        //}
        //
        //inline void switch_internal_energy_mode(std::string name){
        //    sconfig.switch_internal_energy_mode(name);
        //}

        
        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// analysis utilities
        ////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// I/O
        ////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// Simulation control
        ////////////////////////////////////////////////////////////////////////////////////////////

        f64 evolve_once(f64 dt_input, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id);
    

    
    
    
    };

} // namespace shammodels