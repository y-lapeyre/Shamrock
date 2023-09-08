// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shammodels/amr/zeus/modules/ValueLoader.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;




template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_forces() {





    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;



    ValueLoader<Tvec, TgridVec, Tscal> val_load (context, solver_config, storage);
    ComputeField<Tscal> rho_xm = val_load.load_value_with_gz("rho", {-1,0,0}, "rho_xm");
    ComputeField<Tscal> rho_ym = val_load.load_value_with_gz("rho", {0,-1,0}, "rho_ym");
    ComputeField<Tscal> rho_zm = val_load.load_value_with_gz("rho", {0,0,-1}, "rho_zm");

    ComputeField<Tscal> p_xm = val_load.load_value_with_gz(storage.pressure.get(), {-1,0,0}, "p_xm");
    ComputeField<Tscal> p_ym = val_load.load_value_with_gz(storage.pressure.get(), {0,-1,0}, "p_ym");
    ComputeField<Tscal> p_zm = val_load.load_value_with_gz(storage.pressure.get(), {0,0,-1}, "p_zm");

    shamrock::SchedulerUtility utility(scheduler());
    storage.forces.set(utility.make_compute_field<Tvec>("forces", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sycl::buffer<Tscal> & buf_p = storage.pressure.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho   = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        sycl::buffer<Tscal> & buf_rho_xm = rho_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> & buf_rho_ym = rho_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> & buf_rho_zm = rho_zm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> & buf_p_xm = p_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> & buf_p_ym = p_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> & buf_p_zm = p_zm.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor grad_p{forces_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor rho_xm{buf_rho_xm, cgh, sycl::read_only};
            sycl::accessor rho_ym{buf_rho_ym, cgh, sycl::read_only};
            sycl::accessor rho_zm{buf_rho_zm, cgh, sycl::read_only};
            sycl::accessor p{buf_p, cgh, sycl::read_only};
            sycl::accessor p_xm{buf_p_xm, cgh, sycl::read_only};
            sycl::accessor p_ym{buf_p_ym, cgh, sycl::read_only};
            sycl::accessor p_zm{buf_p_zm, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute grad p", [=](u64 id_a) {

                Tvec d_cell = (cell_max[id_a] - cell_min[id_a]).template convert<Tscal>() *
                               coord_conv_fact;

                Tscal rho_i_j_k = rho[id_a];
                Tscal rho_im1_j_k = rho_xm[id_a];
                Tscal rho_i_jm1_k = rho_ym[id_a];
                Tscal rho_i_j_km1 = rho_zm[id_a];

                Tscal p_i_j_k = p[id_a]; 
                Tscal p_im1_j_k = p_xm[id_a];
                Tscal p_i_jm1_k = p_ym[id_a];
                Tscal p_i_j_km1 = p_zm[id_a];

                Tvec dp = {
                    p_i_j_k - p_im1_j_k,
                    p_i_j_k - p_i_jm1_k,
                    p_i_j_k - p_i_j_km1
                };

                Tvec avg_rho = Tvec{
                    rho_i_j_k - rho_im1_j_k,
                    rho_i_j_k - rho_i_jm1_k,
                    rho_i_j_k - rho_i_j_km1
                }*Tscal{0.5};

                Tvec grad_p_source_term = dp/(avg_rho*d_cell);
                
                grad_p[id_a] = grad_p_source_term;
            });
        });
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor forces{forces_buf, cgh, sycl::read_write};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "add ext force", [=](u64 id_a) {

                Tvec block_min = cell_min[id_a].template convert<Tscal>();
                Tvec block_max = cell_max[id_a].template convert<Tscal>();
                Tvec delta_cell = (block_max - block_min)/Block::side_size;
                Tvec delta_cell_h = delta_cell * Tscal(0.5);

                Block::for_each_cell_in_block(delta_cell, [=](u32 lid, Tvec delta){

                    auto get_ext_force = [](Tvec r) {
                        Tscal d = sycl::length(r);
                        return r / (d * d * d);
                    };
                    
                    forces[id_a*Block::block_size + lid] +=  get_ext_force(block_min + delta + delta_cell_h);
                });

            });
        });

        logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });

    
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::apply_force(Tscal dt) {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ivel       = pdl.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &vel_buf    = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor forces{forces_buf, cgh, sycl::read_only};
            sycl::accessor vel{vel_buf, cgh, sycl::read_write};

            shambase::parralel_for(cgh, pdat.get_obj_cnt()*Block::block_size, "add ext force", [=](u64 id_a) {
                vel[id_a] += dt * forces[id_a];
            });
        });

        logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });

    

    storage.forces.reset();
}



template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_AV() {

}

template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;