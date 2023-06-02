// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "SPHModel.hpp"
#include "shambase/exception.hpp"
#include "shamrock/sph/kernels.hpp"

namespace shammodels{

    /**
     * @brief List of the possible models
     * 
     */
    using VariantSPHModelBind = std::variant<
        std::unique_ptr<SPHModel<f64_3, shamrock::sph::kernels::M4>>
    >;


    /**
     * @brief Generic SPH model that can be any kernel/representation
     * 
     */
    class SPHModelVariantBindings{public:
        VariantSPHModelBind var_model;

        SPHModelVariantBindings(ShamrockCtx & ctx, std::string Tvec_name, std::string kernel_name){
            if(Tvec_name == "f64_3" && kernel_name == "M4"){
                var_model = std::make_unique<
                    SPHModel<f64_3, shamrock::sph::kernels::M4>
                >(ctx);
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown combination of representation and kernel");
            }
        }

        inline void init_scheduler(u32 crit_split, u32 crit_merge){
            std::visit([&](auto & arg){
                arg->init_scheduler(crit_split,crit_merge);
            },var_model);
        }

        inline f64 evolve_once(f64 dt_input, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id){
            
            return std::visit([&](auto & arg){
                return arg->evolve_once(dt_input, enable_physics, do_dump,vtk_dump_name,vtk_dump_patch_id);
            },var_model);

        }

        inline void set_cfl_cour(f64 val){
            std::visit([&](auto & arg){
                arg->solver.tmp_solver.set_cfl_cour(val);
            },var_model);
        }

        inline void set_cfl_force(f64 val){
            std::visit([&](auto & arg){
                arg->solver.tmp_solver.set_cfl_force(val);
            },var_model);
        }

        inline void set_particle_mass(f64 val){
            std::visit([&](auto & arg){
                arg->solver.tmp_solver.set_particle_mass(val);
            },var_model);
        }

        inline std::array<f64,3> get_box_dim_fcc_3d(f64 dr, u32 xcnt, u32 ycnt, u32 zcnt){

            std::array<f64,3> ret;

            std::visit([&](auto & arg){

                using SPHModel_t_ptr = typename std::remove_reference<decltype(arg)>::type;
                using SPHModel_t = typename SPHModel_t_ptr::element_type;
                constexpr u32 dim = SPHModel_t::dim;

                if constexpr (dim == 3){
                    auto tmp = arg->get_box_dim_fcc_3d(dr,xcnt,ycnt,zcnt);
                    ret[0] = tmp.x();
                    ret[1] = tmp.y();
                    ret[2] = tmp.z();
                }else{
                    throw shambase::throw_with_loc<std::invalid_argument>(
                        shambase::format("the fcc lattice can only be used in 3d, current dim = {}"
                        ,dim
                        )
                    );
                }

            },var_model);

            return ret;
        }
    };

}