// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGraphGen.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "AMRGraphGen.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::find_AMR_block_graph_links() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    shambase::DistributedData<OrientedAMRBlockGraph> block_graph_links;

    storage.trees.get().for_each([&](u64 id, RTree &tree) {
        u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count           = leaf_count + internal_cell_count;

        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        OrientedAMRBlockGraph result;

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &tree_bmin
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> &tree_bmax
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        for (u32 dir = 0; dir < 6; dir++) {

            TgridVec dir_offset = result.offset_check[dir];

            // [i] is the number of link for block i in mpdat (last value is 0)
            sycl::buffer<u32> link_counts(mpdat.total_elements + 1);

            // fill buffer with number of link in the block graph
            q.submit([&](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator block_looper(tree, cgh);

                sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
                sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};

                sycl::accessor link_cnt{link_counts, cgh, sycl::write_only, sycl::no_init};

                shambase::parralel_for(
                    cgh, mpdat.total_elements, "compute neigh cache 1", [=](u64 gid) {
                        u32 id_a = (u32) gid;

                        shammath::AABB<TgridVec> block_aabb{
                            acc_block_min[id_a], acc_block_max[id_a]};

                        TgridVec block_size = block_aabb.delt();
                        TgridVec block_check_offset
                            = {block_size.x() * dir_offset.x(),
                               block_size.y() * dir_offset.y(),
                               block_size.z() * dir_offset.z()};

                        shammath::AABB<TgridVec> check_aabb{
                            block_aabb.lower + block_check_offset,
                            block_aabb.upper + block_check_offset};

                        u32 block_found_count = 0;

                        block_looper.rtree_for(
                            [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                                return shammath::AABB<TgridVec>{bmin, bmax}
                                    .get_intersect(check_aabb)
                                    .is_volume_not_null();
                            },
                            [&](u32 id_b) {
                                bool interact
                                    = shammath::AABB<
                                          TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                                          .get_intersect(check_aabb)
                                          .is_volume_not_null();

                                if (interact) {
                                    block_found_count++;
                                }
                            });

                        link_cnt[id_a] = block_found_count;
                    });
            });

            // set the last val to 0 so that the last slot after exclusive scan is the sum
            shamalgs::memory::set_element<u32>(q, link_counts, mpdat.total_elements, 0);

            sycl::buffer<u32> link_cnt_offsets
                = shamalgs::numeric::exclusive_sum(q, link_counts, mpdat.total_elements + 1);

            u32 link_cnt
                = shamalgs::memory::extract_element(q, link_cnt_offsets, mpdat.total_elements);

            sycl::buffer<u32> ids_links(link_cnt);

            // find the neigh ids
            q.submit([&](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator block_looper(tree, cgh);

                sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
                sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};

                sycl::accessor cnt_offsets{link_cnt_offsets, cgh, sycl::read_only};
                sycl::accessor links{ids_links, cgh, sycl::write_only, sycl::no_init};

                shambase::parralel_for(
                    cgh, mpdat.total_elements, "compute neigh cache 1", [=](u64 gid) {
                        u32 id_a = (u32) gid;

                        u32 next_link_idx = cnt_offsets[id_a];
                        shammath::AABB<TgridVec> block_aabb{
                            acc_block_min[id_a], acc_block_max[id_a]};

                        TgridVec block_size = block_aabb.delt();
                        TgridVec block_check_offset
                            = {block_size.x() * dir_offset.x(),
                               block_size.y() * dir_offset.y(),
                               block_size.z() * dir_offset.z()};

                        shammath::AABB<TgridVec> check_aabb{
                            block_aabb.lower + block_check_offset,
                            block_aabb.upper + block_check_offset};

                        block_looper.rtree_for(
                            [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                                return shammath::AABB<TgridVec>{bmin, bmax}
                                    .get_intersect(check_aabb)
                                    .is_volume_not_null();
                            },
                            [&](u32 id_b) {
                                bool interact
                                    = shammath::AABB<
                                          TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                                          .get_intersect(check_aabb)
                                          .is_volume_not_null();

                                if (interact) {
                                    links[next_link_idx] = id_b;
                                    next_link_idx++;
                                }
                            });
                    });
            });

            std::unique_ptr<AMRBlockGraph> tmp_graph = std::make_unique<AMRBlockGraph>(
                AMRBlockGraph{std::move(link_cnt_offsets), std::move(ids_links), link_cnt});

            result.block_graph_links[dir] = std::move(tmp_graph);
        }

        block_graph_links.add_obj(id, std::move(result));
    });

}

template class shammodels::basegodunov::modules::AMRGraphGen<f64_3, i64_3>;