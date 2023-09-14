// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/TransportStep.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::TransportStep<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_face_momentas() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    shamrock::SchedulerUtility utility(scheduler());
    storage.pi_xm.set(utility.make_compute_field<Tscal>("pi_xm", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.pi_xp.set(utility.make_compute_field<Tscal>("pi_xp", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.pi_ym.set(utility.make_compute_field<Tscal>("pi_ym", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.pi_yp.set(utility.make_compute_field<Tscal>("pi_yp", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.pi_zm.set(utility.make_compute_field<Tscal>("pi_zm", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.pi_zp.set(utility.make_compute_field<Tscal>("pi_zp", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    ComputeField<Tscal> &pi_xm = storage.pi_xm.get();
    ComputeField<Tscal> &pi_xp = storage.pi_xp.get();
    ComputeField<Tscal> &pi_ym = storage.pi_ym.get();
    ComputeField<Tscal> &pi_yp = storage.pi_yp.get();
    ComputeField<Tscal> &pi_zm = storage.pi_zm.get();
    ComputeField<Tscal> &pi_zp = storage.pi_zp.get();

    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tvec> &buf_vel  = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_pi_xm = pi_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_pi_xp = pi_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_pi_ym = pi_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_pi_yp = pi_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_pi_zm = pi_zm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_pi_zp = pi_zp.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};

            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};

            sycl::accessor pi_xm{buf_pi_xm, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor pi_xp{buf_pi_xp, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor pi_ym{buf_pi_ym, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor pi_yp{buf_pi_yp, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor pi_zm{buf_pi_zm, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor pi_zp{buf_pi_zp, cgh, sycl::write_only, sycl::no_init};

            Block::for_each_cells(cgh,  mpdat.total_elements,"compite Pis", 
                [=](u32 block_id, u32 cell_gid){
                    Tscal r  = rho[cell_gid];
                    Tvec vm  = vel[cell_gid];
                    Tvec vxp = vel_xp[cell_gid];
                    Tvec vyp = vel_yp[cell_gid];
                    Tvec vzp = vel_zp[cell_gid];

                    Tvec tmp_m  = vm * r;
                    Tscal tmp_x = vxp.x() * r;
                    Tscal tmp_y = vyp.y() * r;
                    Tscal tmp_z = vzp.z() * r;

                    pi_xm[cell_gid] = tmp_m.x();
                    pi_ym[cell_gid] = tmp_m.y();
                    pi_zm[cell_gid] = tmp_m.z();
                    pi_xp[cell_gid] = tmp_x;
                    pi_yp[cell_gid] = tmp_y;
                    pi_zp[cell_gid] = tmp_z;
                }
            );

        });
    });
}