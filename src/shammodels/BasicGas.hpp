// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/type_aliases.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph {

    class BasicGas {
        using flt      = f64;
        using vec      = f64_3;
        using u_morton = u32;
        // using Kernel = models::sph::kernels::M4<flt>;

        static constexpr flt htol_up_tol  = 1.4;
        static constexpr flt htol_up_iter = 1.2;

        flt cfl_cour   = -1;
        flt cfl_force  = -1;
        flt gpart_mass = -1;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        public:
        BasicGas(ShamrockCtx &context) : context(context){};

        inline void setup_fields(){
            context.pdata_layout_add_field<vec>("xyz", 1);
            context.pdata_layout_add_field<vec>("vxyz", 1);
            context.pdata_layout_add_field<vec>("axyz", 1);
            context.pdata_layout_add_field<flt>("hpart", 1);
            context.pdata_layout_add_field<flt>("uint", 1);
        }

        inline void check_valid() {
            if (cfl_cour < 0) {
                throw shambase::throw_with_loc<std::invalid_argument>("cfl courant not set");
            }

            if (cfl_force < 0) {
                throw shambase::throw_with_loc<std::invalid_argument>("cfl force not set");
            }

            if (gpart_mass < 0) {
                throw shambase::throw_with_loc<std::invalid_argument>("particle mass not set");
            }
        }

        void dump_vtk(std::string dump_name);

        f64 get_cfl_dt();

        void apply_position_boundary();

        /**
         * @brief 
         * 
         * @param dt 
         */
        void evolve(f64 dt);

        u64 count_particles();

    };

} // namespace shammodels::sph