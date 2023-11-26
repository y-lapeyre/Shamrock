// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExternalForces.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ExternalForces.hpp"

#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamunits/Constants.hpp"

template<class Tvec, class Tmorton, template<class> class SPHKernel>
using Module = shammodels::sph::modules::NeighbourCache<Tvec,Tmorton, SPHKernel>;


template<class Tvec, class Tmorton, template<class> class SPHKernel>
void Module<Tvec,Tmorton, SPHKernel>::start_neighbors_cache() {

    // interface_control
    using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache   = typename GhostHandle::CacheMap;
    using PreStepMergedField = typename GhostHandle::PreStepMergedField;
    using RTree = RadixTree<Tmorton, Tvec>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    storage.neighbors_cache.set(shamrock::tree::ObjectCacheHandler(u64(10e9), [&](u64 patch_id) {
        logger::debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        PreStepMergedField &mfield = storage.merged_xyzh.get().get(patch_id);

        sycl::buffer<Tvec> &buf_xyz    = shambase::get_check_ref(mfield.field_pos.get_buf());
        sycl::buffer<Tscal> &buf_hpart = shambase::get_check_ref(mfield.field_hpart.get_buf());
        sycl::buffer<Tscal> &tree_field_rint = shambase::get_check_ref(
            storage.rtree_rint_field.get().get(patch_id).radix_tree_field_buf);

        sycl::range range_npart{mfield.original_elements};

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);

        u32 obj_cnt       = mfield.original_elements;
        Tscal h_tolerance = solver_config.htol_up_tol;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sycl::buffer<u32> neigh_count(obj_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor neigh_cnt{neigh_count, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor scanned_neigh_cnt{pcache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pcache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });
        });

        return pcache;
    }));

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        storage.neighbors_cache.get().preload(cur_p.id_patch);
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();

}


using namespace shammath;
template class shammodels::sph::modules::NeighbourCache<f64_3,u32, M4>;
template class shammodels::sph::modules::NeighbourCache<f64_3,u32, M6>;