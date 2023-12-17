// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file StencilGenerator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */


#include "shammodels/amr/basegodunov/modules/StencilGenerator.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::fill_slot(i64_3 relative_pos, StencilOffsets result_offset){

    StackEntry stack_loc{};



    using MergedPDat = shamrock::MergedPatchData;
    using RTree = typename Storage::RTree;

    storage.trees.get().for_each([&](u64 id, RTree & tree){
        
        u32 leaf_count = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count = leaf_count + internal_cell_count;

        MergedPDat &mpdat                    = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<TgridVec> & tree_bmin = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> & tree_bmax = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);



        sycl::buffer<u32> stencil_block_idx (stencil_offset_count*mpdat.total_elements);


        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator particle_looper(tree, cgh);



        });

    });

    //sycl::buffer<u32> stencil_block_idx (stencil_offset_count*)

}