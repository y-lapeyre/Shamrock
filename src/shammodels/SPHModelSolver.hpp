// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/BasicGas.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels {
    
    enum InternalEnergyMode{
        /**
        * @brief Disable the Internal energy field
        */
        None,

        /**
        * @brief internal energy equation without artificial viscosity
        */
        NoArtificialViscosity,

        /**
        * @brief Price 2012 u solver with artificial viscosity
        */
        PriceEtAl2012,

        /**
        * @brief Price 2018 phantom solver for artificial viscosity (2012 + averaging of vsig)
        */
        PriceEtAl2018

    };

    //exemple config to show what i would want
    struct Config{

        InternalEnergyMode uint_mode;

        inline void switch_internal_energy_mode(std::string s){
            if(s == "None"){
                //uint_mode = None;
                shambase::throw_unimplemented();
            }else if(s == "NoArtificialViscosity"){
                //uint_mode = NoArtificialViscosity;
                shambase::throw_unimplemented();
            }else if(s == "PriceEtAl2012"){
                uint_mode = PriceEtAl2012;
            }else if(s == "PriceEtAl2018"){
                //uint_mode = PriceEtAl2018;
                shambase::throw_unimplemented();
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown internal energy mode");
            }
        }

        inline  void enable_barotropic(){
            switch_internal_energy_mode("None");
        }

        inline bool has_uint_field(){
            return uint_mode != None;
        }

    };







    /**
     * @brief The shamrock SPH model
     * 
     * @tparam Tvec 
     * @tparam SPHKernel 
     */
    template<class Tvec, template<class> class SPHKernel>
    class SPHModelSolver {public:

        using Tscal                    = shambase::VecComponent<Tvec>;
        static constexpr u32 dimension = shambase::VectorProperties<Tvec>::dimension;
        using Kernel = SPHKernel<Tscal>;

        

        sph::BasicGas tmp_solver; // temporary all of this should be in the solver in fine

        SPHModelSolver(ShamrockCtx &context) : tmp_solver(context){}

        Tscal eos_gamma;
        Tscal gpart_mass;
        Tscal cfl_cour;
        Tscal cfl_force;

        inline void init_required_fields(){
            tmp_solver.setup_fields();
        }

        inline Tscal evolve_once(Tscal dt_input, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id){
            tmp_solver.set_cfl_cour(cfl_cour);
            tmp_solver.set_cfl_force(cfl_force);
            tmp_solver.set_particle_mass(gpart_mass);
            //tmp_solver.set_gamma(eos_gamma);
            
            return tmp_solver.evolve(dt_input, enable_physics, {do_dump,vtk_dump_name,vtk_dump_patch_id});
        }



        





    };

}