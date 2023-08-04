// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::FaceFlagger<Tvec, TgridVec>;

// this flags faces but not your face

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::flag_faces() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<sycl::buffer<u8>> face_normals_dat_lookup;

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> face_normals_lookup(pcache.sum_neigh_cnt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::ObjectCacheIterator cell_looper(pcache, cgh);

            sycl::accessor normals_lookup{
                face_normals_lookup, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "flag_neigh", [=](u64 id_a) {
                TgridVec cell2_a = (cell_min[id_a] + cell_max[id_a]);

                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u64 id_list) {
                    TgridVec cell2_b = (cell_min[id_b] + cell_max[id_b]);
                    TgridVec cell2_d = cell2_b - cell2_a;

                    TgridVec d_norm = sycl::abs(cell2_d).template convert<Tgridscal>();

                    // I mean if you are up to such
                    Tgridscal max_compo = sycl::max(sycl::max(d_norm.x(), d_norm.y()), d_norm.z());

                    // what a readable piece of code
                    // there can be only ONE that is the true answers
                    const u8 lookup = ((cell2_d.x() == max_compo) ? 0 : 0) +
                                      ((cell2_d.x() == -max_compo) ? 1 : 0) +
                                      ((cell2_d.y() == max_compo) ? 2 : 0) +
                                      ((cell2_d.y() == -max_compo) ? 3 : 0) +
                                      ((cell2_d.z() == max_compo) ? 4 : 0) +
                                      ((cell2_d.z() == -max_compo) ? 5 : 0);

                    // F this bit bit of code
                    // i'm so done with this crap
                    // godbolts gods command's you to inline !
                    normals_lookup[id_list] = lookup;
                });

                // Chaptgpt beautifull poem about the beautifullness of the SIMD instructions
                //
                // Oh, SIMD instructions, you tangled mess,
                // A source of frustration, I must confess.
                // You promised speed, you boasted grace,
                // Yet you leave my code a tangled case.
                //
                // With your cryptic syntax and obscure ways,
                // You lead me into a bewildering maze.
                // I try to optimize, to harness your might,
                // But your convoluted logic gives me a fright.
                //
                // You claim to be efficient, a boon to behold,
                // Yet your pitfalls and traps leave me cold.
                // I chase after vectors, I chase after speed,
                // But your complexities multiply with every need.
                //
                // Oh, SIMD instructions, you deceptive charm,
                // You leave my patience disarmed.
                // A Pandora's box of headaches and woes,
                // In your shadowy realm, my confidence slows.
                //
                // So here's to you, SIMD, with a bitter disdain,
                // Your alluring facade hides nothing but pain.
                // You promised elegance, you promised glee,
                // But all I find is chaos, as you laugh at me.
            });
        });

        // store the buffer in distrib data
        face_normals_dat_lookup.add_obj(p.id_patch, std::move(face_normals_lookup));
    });

    storage.face_normals_lookup.set(std::move(face_normals_dat_lookup));
}

template class shammodels::zeus::modules::FaceFlagger<f64_3, i64_3>;