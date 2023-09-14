// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/TransportStep.hpp"
#include "shammodels/amr/zeus/modules/ValueLoader.hpp"
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
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    shamrock::SchedulerUtility utility(scheduler());
    storage.Q.set(utility.make_compute_field<sycl::vec<Tscal,8>>("Q", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    ComputeField<sycl::vec<Tscal,8>> & Q   = storage.Q  .get();

    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &buf_vel  = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);
        
        sycl::buffer<sycl::vec<Tscal,8>> &buf_Q   = Q  .get_buf_check(p.id_patch);    

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            sycl::accessor eint{buf_eint, cgh, sycl::read_only};

            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};
            
            sycl::accessor Q   {buf_Q  , cgh, sycl::write_only,sycl::no_init};

            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite Pis", [=](u32 /*block_id*/, u32 cell_gid) {
                    Tscal r  = rho[cell_gid];
                    Tscal e = eint[cell_gid];
                    Tvec vm  = vel[cell_gid];
                    Tvec vxp = vel_xp[cell_gid];
                    Tvec vyp = vel_yp[cell_gid];
                    Tvec vzp = vel_zp[cell_gid];

                    Tvec tmp_m  = vm * r;
                    Tscal tmp_x = vxp.x() * r;
                    Tscal tmp_y = vyp.y() * r;
                    Tscal tmp_z = vzp.z() * r;

                    Q[cell_gid] = {r,
                        tmp_m.x(),
                        tmp_m.y(),
                        tmp_m.z(),
                        tmp_x,
                        tmp_y,
                        tmp_z,
                        e};
                });
        });
    });
}


template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_dq() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    using Tscal8 = sycl::vec<Tscal,8>;
    ComputeField<Tscal8> &Q = storage.Q.get();

    modules::ValueLoader<Tvec, TgridVec, Tscal8> val_load_vec8(context, solver_config, storage);
    
    storage.dQ_x.set( val_load_vec8.load_value_with_gz(Q, {-1, 0, 0}, "dQ_x"));
    storage.dQ_y.set( val_load_vec8.load_value_with_gz(Q, {0, -1, 0}, "dQ_y"));
    storage.dQ_z.set( val_load_vec8.load_value_with_gz(Q, {0, 0, -1}, "dQ_z"));

    
    ComputeField<Tscal8> &dQ_x = storage.dQ_x.get();
    ComputeField<Tscal8> &dQ_y = storage.dQ_y.get();
    ComputeField<Tscal8> &dQ_z = storage.dQ_z.get();


}


template class shammodels::zeus::modules::TransportStep<f64_3, i64_3>;