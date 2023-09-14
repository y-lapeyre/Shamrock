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
void Module<Tvec, TgridVec>::compute_cell_centered_momentas() {
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
    storage.Q.set(
        utility.make_compute_field<sycl::vec<Tscal, 8>>("Q", Block::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        }));

    ComputeField<sycl::vec<Tscal, 8>> &Q = storage.Q.get();

    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &buf_vel   = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q = Q.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            sycl::accessor eint{buf_eint, cgh, sycl::read_only};

            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::write_only, sycl::no_init};

            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite Pis", [=](u32 /*block_id*/, u32 cell_gid) {
                    Tscal r  = rho[cell_gid];
                    Tscal e  = eint[cell_gid];
                    Tvec vm  = vel[cell_gid];
                    Tvec vxp = vel_xp[cell_gid];
                    Tvec vyp = vel_yp[cell_gid];
                    Tvec vzp = vel_zp[cell_gid];

                    Tvec tmp_m  = vm * r;
                    Tscal tmp_x = vxp.x() * r;
                    Tscal tmp_y = vyp.y() * r;
                    Tscal tmp_z = vzp.z() * r;

                    Q[cell_gid] = {r, tmp_m.x(), tmp_m.y(), tmp_m.z(), tmp_x, tmp_y, tmp_z, e};
                });
        });
    });
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_limiter() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    using Tscal8            = sycl::vec<Tscal, 8>;
    ComputeField<Tscal8> &Q = storage.Q.get();

    modules::ValueLoader<Tvec, TgridVec, Tscal8> val_load_vec8(context, solver_config, storage);

    

    storage.Q_xm.set(val_load_vec8.load_value_with_gz(Q, {-1, 0, 0}, "Q_xm"));
    storage.Q_ym.set(val_load_vec8.load_value_with_gz(Q, {0, -1, 0}, "Q_ym"));
    storage.Q_zm.set(val_load_vec8.load_value_with_gz(Q, {0, 0, -1}, "Q_zm"));

    ComputeField<Tscal8> & Q_xm = storage.Q_xm.get();
    ComputeField<Tscal8> & Q_ym = storage.Q_ym.get();
    ComputeField<Tscal8> & Q_zm = storage.Q_zm.get();

    ComputeField<Tscal8> Q_xp = val_load_vec8.load_value_with_gz(Q, {1, 0, 0}, "Q_xp");
    ComputeField<Tscal8> Q_yp = val_load_vec8.load_value_with_gz(Q, {0, 1, 0}, "Q_yp");
    ComputeField<Tscal8> Q_zp = val_load_vec8.load_value_with_gz(Q, {0, 0, 1}, "Q_zp");

    shamrock::SchedulerUtility utility(scheduler());

    storage.a_x.set(utility.make_compute_field<Tscal8>("a_x", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.a_y.set(utility.make_compute_field<Tscal8>("a_y", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.a_z.set(utility.make_compute_field<Tscal8>("a_z", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    ComputeField<Tscal8> & a_x =storage.a_z.get();
    ComputeField<Tscal8> & a_y =storage.a_z.get();
    ComputeField<Tscal8> & a_z =storage.a_z.get();

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat                    = storage.merged_patchdata_ghost.get().get(p.id_patch);
        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q     = Q.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_xm = Q_xm.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_xp = Q_xp.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_x   = a_x.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_ym = Q_ym.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_yp = Q_yp.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_y   = a_y.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_zm = Q_zm.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_zp = Q_zp.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_z   = a_z.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_xm{buf_Q_xm, cgh, sycl::read_only};
            sycl::accessor Q_xp{buf_Q_xp, cgh, sycl::read_only};
            sycl::accessor a_x{buf_a_x, cgh, sycl::write_only, sycl::no_init};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;

            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_x", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().x() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_xm[cell_gid];
                    Tscal8 Qip = Q_xp[cell_gid];

                    Tscal8 dqm = (Qi - Qim) / d_cell;
                    Tscal8 dqp = (Qip - Qi) / d_cell;

                    auto vanleer = [](Tscal8 f, Tscal8 g) -> Tscal8 {
                        Tscal8 tmp = f * g;
                        tmp        = tmp + shambase::sycl_utils::g_sycl_abs(tmp);
                        return tmp / (f + g);
                    };
                    a_x[cell_gid] = vanleer(dqm, dqp);
                });
        });

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_ym{buf_Q_ym, cgh, sycl::read_only};
            sycl::accessor Q_yp{buf_Q_yp, cgh, sycl::read_only};
            sycl::accessor a_y{buf_a_y, cgh, sycl::write_only, sycl::no_init};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;

            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_y", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().y() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_ym[cell_gid];
                    Tscal8 Qip = Q_yp[cell_gid];

                    Tscal8 dqm = (Qi - Qim) / d_cell;
                    Tscal8 dqp = (Qip - Qi) / d_cell;

                    auto vanleer = [](Tscal8 f, Tscal8 g) -> Tscal8 {
                        Tscal8 tmp = f * g;
                        tmp        = tmp + shambase::sycl_utils::g_sycl_abs(tmp);
                        return tmp / (f + g);
                    };
                    a_y[cell_gid] = vanleer(dqm, dqp);
                });
        });

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_zm{buf_Q_zm, cgh, sycl::read_only};
            sycl::accessor Q_zp{buf_Q_zp, cgh, sycl::read_only};
            sycl::accessor a_z{buf_a_z, cgh, sycl::write_only, sycl::no_init};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;

            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_z", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().z() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_zm[cell_gid];
                    Tscal8 Qip = Q_zp[cell_gid];

                    Tscal8 dqm = (Qi - Qim) / d_cell;
                    Tscal8 dqp = (Qip - Qi) / d_cell;

                    auto vanleer = [](Tscal8 f, Tscal8 g) -> Tscal8 {
                        Tscal8 tmp = f * g;
                        tmp        = tmp + shambase::sycl_utils::g_sycl_abs(tmp);
                        return tmp / (f + g);
                    };
                    a_z[cell_gid] = vanleer(dqm, dqp);
                });
        });
    });

    
}



template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_face_centered_moments(Tscal dt_in){

    
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    using Tscal8            = sycl::vec<Tscal, 8>;
    

    modules::ValueLoader<Tvec, TgridVec, Tscal8> val_load_vec8(context, solver_config, storage);
    ComputeField<Tscal8> &Q = storage.Q.get();
    ComputeField<Tscal8> & a_x =storage.a_z.get();
    ComputeField<Tscal8> & a_y =storage.a_z.get();
    ComputeField<Tscal8> & a_z =storage.a_z.get();

    ComputeField<Tscal8> a_xm = val_load_vec8.load_value_with_gz(a_x, {1, 0, 0}, "a_xm");
    ComputeField<Tscal8> a_ym = val_load_vec8.load_value_with_gz(a_y, {1, 0, 0}, "a_ym");
    ComputeField<Tscal8> a_zm = val_load_vec8.load_value_with_gz(a_z, {1, 0, 0}, "a_zm");


    ComputeField<Tscal8> & Q_xm = storage.Q_xm.get();
    ComputeField<Tscal8> & Q_ym = storage.Q_ym.get();
    ComputeField<Tscal8> & Q_zm = storage.Q_zm.get();



    shamrock::SchedulerUtility utility(scheduler());
    storage.Qstar_x.set(utility.make_compute_field<Tscal8>("Qstar_x", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.Qstar_y.set(utility.make_compute_field<Tscal8>("Qstar_y", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.Qstar_z.set(utility.make_compute_field<Tscal8>("Qstar_z", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));


    ComputeField<Tscal8> & Qstar_x = storage.Qstar_x.get();
    ComputeField<Tscal8> & Qstar_y = storage.Qstar_y.get();
    ComputeField<Tscal8> & Qstar_z = storage.Qstar_z.get();


    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat                    = storage.merged_patchdata_ghost.get().get(p.id_patch);
        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q     = Q.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_xm = Q_xm.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_x   = a_x.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_xm   = a_xm.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_ym = Q_ym.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_y   = a_y.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_ym   = a_ym.get_buf_check(p.id_patch);

        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Q_zm = Q_zm.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_z   = a_z.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_a_zm   = a_zm.get_buf_check(p.id_patch);


        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Qstar_x = Qstar_x.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Qstar_y = Qstar_y.get_buf_check(p.id_patch);
        sycl::buffer<sycl::vec<Tscal, 8>> &buf_Qstar_z = Qstar_z.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &buf_vel   = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_xm{buf_Q_xm, cgh, sycl::read_only};
            sycl::accessor a_x{buf_a_x, cgh, sycl::read_only};
            sycl::accessor a_xm{buf_a_xm, cgh, sycl::read_only};
            sycl::accessor Qstar_x{buf_Qstar_x, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor vel{buf_vel, cgh, sycl::read_only};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;
            Tscal dt = dt_in;
            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_z", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().x() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_xm[cell_gid];
                    Tscal8 ai  = a_x[cell_gid];
                    Tscal8 aim = a_xm[cell_gid];
                    Tscal vx = vel[cell_gid].x();

                    Tscal8 res ;

                    if(vx >= 0){
                        res = Qim + aim*(d_cell - vx*dt)*0.5;
                    }else{
                        res = Qi + ai*(d_cell + vx*dt)*0.5;
                    }

                    Qstar_x[cell_gid] = res;
                });
        });


        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_ym{buf_Q_ym, cgh, sycl::read_only};
            sycl::accessor a_y{buf_a_y, cgh, sycl::read_only};
            sycl::accessor a_ym{buf_a_ym, cgh, sycl::read_only};
            sycl::accessor Qstar_y{buf_Qstar_y, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor vel{buf_vel, cgh, sycl::read_only};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;
            Tscal dt = dt_in;
            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_z", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().y() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_ym[cell_gid];
                    Tscal8 ai  = a_y[cell_gid];
                    Tscal8 aim = a_ym[cell_gid];
                    Tscal vy = vel[cell_gid].y();

                    Tscal8 res ;

                    if(vy >= 0){
                        res = Qim + aim*(d_cell - vy*dt)*0.5;
                    }else{
                        res = Qi + ai*(d_cell + vy*dt)*0.5;
                    }

                    Qstar_y[cell_gid] = res;
                });
        });


        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor Q{buf_Q, cgh, sycl::read_only};
            sycl::accessor Q_zm{buf_Q_zm, cgh, sycl::read_only};
            sycl::accessor a_z{buf_a_z, cgh, sycl::read_only};
            sycl::accessor a_zm{buf_a_zm, cgh, sycl::read_only};
            sycl::accessor Qstar_z{buf_Qstar_z, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor vel{buf_vel, cgh, sycl::read_only};

            Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact / Block::block_size;
            Tscal dt = dt_in;
            Block::for_each_cells(
                cgh, mpdat.total_elements, "compite a_z", [=](u32 block_id, u32 cell_gid) {
                    Tscal d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>().z() *
                        coord_conv_fact;

                    Tscal8 Qi  = Q[cell_gid];
                    Tscal8 Qim = Q_zm[cell_gid];
                    Tscal8 ai  = a_z[cell_gid];
                    Tscal8 aim = a_zm[cell_gid];
                    Tscal vz = vel[cell_gid].z();

                    Tscal8 res ;

                    if(vz >= 0){
                        res = Qim + aim*(d_cell - vz*dt)*0.5;
                    }else{
                        res = Qi + ai*(d_cell + vz*dt)*0.5;
                    }

                    Qstar_z[cell_gid] = res;
                });
        });

    });

}

template class shammodels::zeus::modules::TransportStep<f64_3, i64_3>;