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
#include "shammath/CoordRange.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/serialpatchtree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/sph/kernels.hpp"

namespace shammodels::sph {

    class BasicGas {
        using flt      = f64;
        using vec      = f64_3;
        using u_morton = u32;
        using Kernel = shamrock::sph::kernels::M4<flt>;

        static constexpr flt Rkern = Kernel::Rkern;
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
            context.pdata_layout_add_field<flt>("duint", 1);
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


        f64 get_cfl_dt();

        void apply_position_boundary(SerialPatchTree<vec> & sptree);

        struct DumpOption{
            bool vtk_do_dump;
            std::string vtk_dump_fname;
            bool vtk_dump_patch_id;
        };


        /**
         * @brief 
         * 
         * @param dt 
         */
        void evolve(f64 dt, DumpOption dump_opt);

        u64 count_particles();

    };

} // namespace shammodels::sph