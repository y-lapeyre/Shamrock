// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris
// <timothee.david--cleris@ens-lyon.fr> Licensed under CeCILL 2.1 License, see
// LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file StencilGenerator.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/StencilGenerator.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shammodels/amr/AMRBlockStencil.hpp"
#include "shammodels/amr/AMRCellStencil.hpp"

template <class Tvec, class TgridVec, class Objiter>
void _kernel(u32 id_a, Objiter cell_looper, TgridVec const *cell_min,
             TgridVec const *cell_max,
             shammodels::amr::block::StencilElement *stencil_info) {

  namespace block = shammodels::amr::block;

  block::StencilElement ret;

  shammath::AABB<TgridVec> cell_aabb{cell_min[id_a], cell_max[id_a]};

  std::array<u32, 8> found_cells = {u32_max};
  u32 cell_found_count = 0;

  cell_looper.rtree_for(
      [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
        return shammath::AABB<TgridVec>{bmin, bmax}
            .get_intersect(cell_aabb)
            .is_volume_not_null();
      },
      [&](u32 id_b) {
        bool interact = shammath::AABB<TgridVec>{cell_min[id_b], cell_max[id_b]}
                            .get_intersect(cell_aabb)
                            .is_volume_not_null();

        if (interact) {
          if (cell_found_count < 8) {
            found_cells[cell_found_count] = id_b;
          }
          cell_found_count++;
        }
      });

  auto cell_found_0_aabb = shammath::AABB<TgridVec>{cell_min[found_cells[0]],
                                                    cell_max[found_cells[0]]};

  // delt is the linear delta, so it's a factor 2 instead of 8
  bool check_levelp1 =
      shambase::vec_equals(cell_found_0_aabb.delt() * 2, cell_aabb.delt()) &&
      (cell_found_count == 8);

  bool check_levelm1 =
      shambase::vec_equals(cell_found_0_aabb.delt(), cell_aabb.delt() * 2) &&
      (cell_found_count == 1);

  bool check_levelsame =
      shambase::vec_equals(cell_found_0_aabb.delt(), cell_aabb.delt()) &&
      (cell_found_count == 1);

  i32 state =
      i32(check_levelsame) + i32(check_levelm1) * 2 + i32(check_levelp1) * 4;

  switch (state) {
  case 1:
    ret = block::StencilElement(block::SameLevel{found_cells[0]});
    break;
  case 2:
    ret = block::StencilElement(block::Levelm1{found_cells[0]});
    break;

  // for case 4 (level p1) sort elements in corect order
  // cf in block cell lowering
  // u32 child_select = mod_coord[0]*4 + mod_coord[1]*2 + mod_coord[2];
  case 4:
    ret = block::StencilElement(block::Levelp1{found_cells});
    break;
  }

  stencil_info[id_a] = ret;
}

template <class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<
    Tvec, TgridVec>::compute_block_stencil_slot(i64_3 relative_pos,
                                                StencilOffsets result_offset)
    -> dd_block_stencil_el_buf {

  StackEntry stack_loc{};

  using MergedPDat = shamrock::MergedPatchData;
  using RTree = typename Storage::RTree;

  shambase::DistributedData<sycl::buffer<amr::block::StencilElement>>
      block_stencil_element;

  storage.trees.get().for_each([&](u64 id, RTree &tree) {
    u32 leaf_count = tree.tree_reduced_morton_codes.tree_leaf_count;
    u32 internal_cell_count = tree.tree_struct.internal_cell_count;
    u32 tot_count = leaf_count + internal_cell_count;

    MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

    sycl::buffer<TgridVec> &tree_bmin =
        shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
    sycl::buffer<TgridVec> &tree_bmax =
        shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

    sycl::buffer<amr::block::StencilElement> stencil_block_info(
        stencil_offset_count * mpdat.total_elements);

    sycl::buffer<TgridVec> &buf_cell_min =
        mpdat.pdat.get_field_buf_ref<TgridVec>(0);
    sycl::buffer<TgridVec> &buf_cell_max =
        mpdat.pdat.get_field_buf_ref<TgridVec>(1);

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
      shamrock::tree::ObjectIterator cell_looper(tree, cgh);

      sycl::accessor acc_cell_min{buf_cell_min, cgh, sycl::read_only};
      sycl::accessor acc_cell_max{buf_cell_max, cgh, sycl::read_only};

      sycl::accessor acc_stencil_info{stencil_block_info, cgh, sycl::write_only,
                                      sycl::no_init};

      shambase::parralel_for(
          cgh, mpdat.total_elements, "compute neigh cache 1", [=](u64 gid) {
            u32 id_a = (u32)gid;

            _kernel<Tvec, TgridVec>(
                id_a, cell_looper, acc_cell_min.get_pointer(),
                acc_cell_max.get_pointer(), acc_stencil_info.get_pointer());
          });
    });

    block_stencil_element.add_obj(id, std::move(stencil_block_info));
  });

  return block_stencil_element;

}

template <class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::
    lower_block_slot_to_cell(i64_3 relative_pos, StencilOffsets result_offset,
                             dd_block_stencil_el_buf &block_stencil_el)
        -> dd_cell_stencil_el_buf {

  return block_stencil_el.map<cell_stencil_el_buf>(
      [&](u64 id, block_stencil_el_buf &block_stencil_el_b) {
        return lower_block_slot_to_cell(relative_pos, result_offset,
                                        block_stencil_el_b);
      });
}

template <class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::
    lower_block_slot_to_cell(i64_3 relative_pos, StencilOffsets result_offset,
                             block_stencil_el_buf &block_stencil_el)
        -> cell_stencil_el_buf 
{

    sycl::buffer<amr::cell::StencilElement> ret (block_stencil_el.size());

    return ret;
}

template class shammodels::basegodunov::modules::StencilGenerator<f64_3, i64_3>;