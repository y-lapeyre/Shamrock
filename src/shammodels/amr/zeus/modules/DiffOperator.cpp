// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shammodels/amr/zeus/modules/DiffOperator.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::DiffOperator<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_gradu() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    shamrock::SchedulerUtility utility(scheduler());
    storage.gradu.set(utility.make_compute_field<Tvec>("gradu", 3, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> &face_normals_lookup = storage.face_normals_lookup.get().get(p.id_patch);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sycl::buffer<Tvec> &buf_vel  = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);
        sycl::buffer<Tvec> &buf_grad_u = storage.gradu.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::ObjectCacheIterator cell_looper(pcache, cgh);

            sycl::accessor normals_lookup{face_normals_lookup, cgh, sycl::read_only};
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor grad_u{buf_grad_u, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "subsetp1", [=](u64 id_a) {
                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>() *
                               coord_conv_fact * 0.5f;

               // Tvec sum_grad_ux = {};
               // Tvec sum_grad_ux = {};
//
               // // looks like it's on the double preicision roofline there is
               // // nothing to optimize here turn around
               // cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u32 id_list) {
               //     Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>() *
               //                    coord_conv_fact * 0.5f;
//
               //     Tvec n        = Flagger::lookup_to_normal(normals_lookup[id_list]);
               //     Tscal dr_proj = sycl::dot(cell2_b - cell2_a, n);
//
               //     Tvec drm1_n = n / dr_proj;
//
               //     sum_grad_p += drm1_n * rho[id_b];
               // });
//
               // grad_p[id_a] = -sum_grad_p;
            });
        });
    });

}

template class shammodels::zeus::modules::DiffOperator<f64_3, i64_3>;