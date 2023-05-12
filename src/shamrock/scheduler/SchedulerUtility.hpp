// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/memory.hpp"
#include "shambase/sycl.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shamrock/math/integrators.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock {

    class SchedulerUtility {
        PatchScheduler &sched;

        public:

        SchedulerUtility(PatchScheduler &sched) : sched(sched) {}

        template<class T, class flt>
        inline void fields_forward_euler(u32 field_idx, u32 derfield_idx, flt dt) {
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                integrators::forward_euler(
                    shamsys::instance::get_compute_queue(),
                    shambase::get_check_ref(pdat.get_field<T>(field_idx).get_buf()),
                    shambase::get_check_ref(pdat.get_field<T>(derfield_idx).get_buf()),
                    pdat.get_obj_cnt(),
                    dt);
            });
        }

        template<class T, class flt>
        inline void
        fields_leapfrog_corrector(u32 field_idx, u32 derfield_idx, u32 derfield_old_idx, flt hdt) {
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                integrators::leapfrog_corrector(
                    shamsys::instance::get_compute_queue(),
                    shambase::get_check_ref(pdat.get_field<T>(field_idx).get_buf()),
                    shambase::get_check_ref(pdat.get_field<T>(derfield_idx).get_buf()),
                    shambase::get_check_ref(pdat.get_field<T>(derfield_old_idx).get_buf()),
                    pdat.get_obj_cnt(),
                    hdt);
            });
        }

        template<class T>
        inline void fields_apply_periodicity(u32 field_idx, std::pair<T, T> box) {
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                utilities::sycl_position_modulo(
                    shamsys::instance::get_compute_queue(),
                    pdat.get_obj_cnt(),
                    shambase::get_check_ref(pdat.get_field<T>(field_idx).get_buf()),
                    box);
            });
        }

        template<class T>
        inline void fields_swap(u32 field_idx1, u32 field_idx2) {
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                utilities::swap_fields(
                    shamsys::instance::get_compute_queue(),
                    shambase::get_check_ref(pdat.get_field<T>(field_idx1).get_buf()),
                    shambase::get_check_ref(pdat.get_field<T>(field_idx2).get_buf()),
                    pdat.get_obj_cnt());
            });
        }
    };

} // namespace shamrock