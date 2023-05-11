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

    class BasicGas{
        using flt = f64;
        using vec = f64_3;
        using u_morton = u32;
        //using Kernel = models::sph::kernels::M4<flt>;

        static constexpr flt htol_up_tol  = 1.4;
        static constexpr flt htol_up_iter = 1.2;

        flt cfl_cour  = -1;
        flt cfl_force = -1;
        flt gpart_mass = -1;

        ShamrockCtx & context;
        inline PatchScheduler & scheduler(){
            return shambase::get_check_ref(context.sched);
        }

        public:

        BasicGas(ShamrockCtx & context) : context(context){};

        inline void check_valid(){
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

        inline void dump_vtk(std::string dump_name){

            shamrock::LegacyVtkWritter writer(dump_name, true,shamrock::UnstructuredGrid);

            using namespace shamrock::patch;

            u64 num_obj = 0; // TODO get_rank_count() in scheduler
            scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                num_obj += pdat.get_obj_cnt();
            });

            //TODO aggregate field ?
            sycl::buffer<vec> pos(num_obj);

            u64 ptr = 0; // TODO accumulate_field() in scheduler ?
            scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {

                using namespace shamalgs::memory;
                using namespace shambase;

                write_with_offset_into(pos, get_check_ref(pdat.get_field<vec>(0).get_buf()), ptr, pdat.get_obj_cnt());

                ptr += pdat.get_obj_cnt();
            });

            writer.write_points(pos, num_obj);
            
        }

    };

}