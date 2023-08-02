// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/AMRTree.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::AMRTree<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::build_trees(){

    StackEntry stack_loc{};

    using MergedPDat = shambase::DistributedData<shamrock::MergedPatchData>;
    using RTree = typename Storage::RTree;

    MergedPDat & mpdat = storage.merged_patchdata_ghost.get();

    shamrock::patch::PatchDataLayout & mpdl = storage.ghost_layout.get();

    u32 reduc_level = 0;

    storage.merge_patch_bounds.set(
        mpdat.map<shammath::AABB<TgridVec>>([&](u64 id, shamrock::MergedPatchData &merged) {

            logger::debug_ln("AMR", "compute bound merged patch",id);
            
            TgridVec min_bound = merged.pdat.get_field<TgridVec>(0).compute_min();
            TgridVec max_bound = merged.pdat.get_field<TgridVec>(1).compute_max();

            return shammath::AABB<TgridVec>{min_bound,max_bound};
        }));

    shambase::DistributedData<shammath::AABB<TgridVec>> & bounds = storage.merge_patch_bounds.get();

    shambase::DistributedData<RTree> trees =
        mpdat.map<RTree>([&](u64 id, shamrock::MergedPatchData &merged) {

            logger::debug_ln("AMR", "compute tree for merged patch",id);

            auto aabb = bounds.get(id);

            TgridVec bmin = aabb.lower;
            TgridVec bmax = aabb.upper;

            TgridVec diff = bmax - bmin;
            diff.x() = shambase::roundup_pow2(diff.x());
            diff.y() = shambase::roundup_pow2(diff.y());
            diff.z() = shambase::roundup_pow2(diff.z());
            bmax = bmin + diff;

            auto & field_pos = merged.pdat.get_field<TgridVec>(0);

            RTree tree(shamsys::instance::get_compute_queue(),
                        {bmin, bmax},
                        field_pos.get_buf(),
                        field_pos.get_obj_cnt(),
                        reduc_level);

            return tree;
        });

    storage.trees.set(std::move(trees));

}


template class shammodels::zeus::modules::AMRTree<f64_3, i64_3>;