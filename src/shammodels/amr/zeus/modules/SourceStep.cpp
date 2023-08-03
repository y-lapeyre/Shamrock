// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::substep_1(){


    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf      = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf     = ghost_layout.get_field_idx<Tscal>("eint");



    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        tree::ObjectCache & pcache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> & face_normals_lookup = storage.face_normals_lookup.get().get(p.id_patch);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sycl::buffer<Tscal> & buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> grad_rho_x (pdat.get_obj_cnt());

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            tree::ObjectCacheIterator cell_looper(pcache, cgh);

            sycl::accessor normals_lookup{
                face_normals_lookup, cgh, sycl::read_only};
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor rho_grad{grad_rho_x,cgh,sycl::write_only,sycl::no_init};
            sycl::accessor rho{buf_rho,cgh,sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "subsetp1", [=](u64 id_a) {

                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>()*coord_conv_fact*0.5f;

                Tscal gradx_rho = 0;

                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u32 id_list) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>()*coord_conv_fact*0.5f;
                    
                    Tvec n = Flagger::lookup_to_normal(normals_lookup[id_list]);
                    Tscal dr_proj = sycl::dot(cell2_b - cell2_a, n);

                    Tvec drm1_n = n/dr_proj;

                    gradx_rho += sycl::dot(drm1_n, Tvec{1,0,0})*rho[id_b];

                });

                rho_grad[id_a] = gradx_rho;

            });
            

        });

    });

}


template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;