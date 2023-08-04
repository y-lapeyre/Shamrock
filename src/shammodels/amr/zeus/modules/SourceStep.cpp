// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_forces() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    shamrock::SchedulerUtility utility(scheduler());
    storage.forces.set(utility.make_compute_field<Tvec>("forces", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> &face_normals_lookup = storage.face_normals_lookup.get().get(p.id_patch);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sycl::buffer<Tscal> &buf_rho   = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::ObjectCacheIterator cell_looper(pcache, cgh);

            sycl::accessor normals_lookup{face_normals_lookup, cgh, sycl::read_only};
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor grad_p{forces_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "subsetp1", [=](u64 id_a) {
                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>() *
                               coord_conv_fact * 0.5f;

                Tvec sum_grad_p = {};

                // looks like it's on the double preicision roofline there is
                // nothing to optimize here turn around
                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u32 id_list) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>() *
                                   coord_conv_fact * 0.5f;

                    Tvec n        = Flagger::lookup_to_normal(normals_lookup[id_list]);
                    Tscal dr_proj = sycl::dot(cell2_b - cell2_a, n);

                    Tvec drm1_n = n / dr_proj;

                    sum_grad_p += drm1_n * rho[id_b];
                });

                grad_p[id_a] = -sum_grad_p;
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
                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>() *
                               coord_conv_fact * 0.5f;

                auto get_ext_force = [](Tvec r) {
                    Tscal d = sycl::length(r);
                    return r / (d * d * d);
                };

                forces[id_a] += (forces[id_a] / rho[id_a]) + get_ext_force(cell2_a);
            });
        });
    });
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::apply_force(Tscal dt) {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ivel       = pdl.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &vel_buf    = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor forces{forces_buf, cgh, sycl::read_only};
            sycl::accessor vel{vel_buf, cgh, sycl::read_write};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "add ext force", [=](u64 id_a) {
                vel[id_a] += dt * forces[id_a];
            });
        });
    });

    storage.forces.reset();
}

template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;