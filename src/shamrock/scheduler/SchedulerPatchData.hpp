// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file scheduler_patch_data.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief PatchData handling
 * @version 0.1
 * @date 2022-03-01
 *
 * @copyright Copyright (c) 2022
 *
 */


#include <cmath>
#include <map>

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/Patch.hpp"

#include "shamrock/legacy/patch/scheduler/scheduler_patch_list.hpp"
#include "shamrock/legacy/patch/sim_box.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"

/**
 * @brief Class to handle PatchData owned by the node
 *
 */
class SchedulerPatchData {
    public:
    shamrock::patch::PatchDataLayout &pdl;

    /**
     * @brief map container for patchdata owned by the current node (layout : id_patch,data)
     *
     */
    std::map<u64, shamrock::patch::PatchData> owned_data;

    inline bool has_patch(u64 id){
        return owned_data.find(id) != owned_data.end();
    }

    inline shamrock::patch::PatchData & get_pdat(u64 id){
        return owned_data.at(id);
    }

    /**
     * @brief simulation box geometry info
     *
     */
    shamrock::patch::SimulationBoxInfo sim_box;

    /**
     * @brief apply a load balancing change list to shuffle patchdata arround the cluster
     *
     * //TODO clean this documentation
     *
     * @param change_list
     * @param patch_list
     */
    void apply_change_list(
        const shamrock::scheduler::LoadBalancingChangeList & change_list, SchedulerPatchList &patch_list
    );

    /**
     * @brief split a patchdata into 8 childs according to the 8 patches in arguments
     *
     * @param key_orginal key of the original patchdata
     * @param p0
     * @param p1
     * @param p2
     * @param p3
     * @param p4
     * @param p5
     * @param p6
     * @param p7
     */
    void split_patchdata(u64 key_orginal, const std::array<shamrock::patch::Patch, 8> patches);

    /**
     * @brief merge 8 old patchdata into one
     *
     * @param new_key new key to store the merge data in the map
     * @param old_key0
     * @param old_key1
     * @param old_key2
     * @param old_key3
     * @param old_key4
     * @param old_key5
     * @param old_key6
     * @param old_key7
     */
    void merge_patchdata(
        u64 new_key,
        u64 old_key0,
        u64 old_key1,
        u64 old_key2,
        u64 old_key3,
        u64 old_key4,
        u64 old_key5,
        u64 old_key6,
        u64 old_key7
    );

    inline SchedulerPatchData(
        shamrock::patch::PatchDataLayout &pdl, shamrock::patch::PatchCoord patch_coord_range)
         : pdl(pdl), sim_box(pdl,patch_coord_range) {}
};