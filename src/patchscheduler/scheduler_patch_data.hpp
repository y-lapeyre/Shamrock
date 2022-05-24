// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

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

#pragma once

#include <cmath>
#include <map>

#include "patch/patch.hpp"
#include "patch/patchdata.hpp"

#include "patch/patchdata_layout.hpp"
#include "scheduler_patch_list.hpp"
#include "sim_box.hpp"

/**
 * @brief Class to handle PatchData owned by the node
 * 
 */
class SchedulerPatchData {
  public:

    PatchDataLayout & pdl;

    /**
     * @brief map container for patchdata owned by the current node (layout : id_patch,data)
     * 
     */
    std::map<u64, PatchData> owned_data;

    /**
     * @brief simulation box geometry info
     * 
     */
    SimulationBoxInfo sim_box;

    /**
     * @brief apply a load balancing change list to shuffle patchdata arround the cluster
     * 
     * //TODO clean this documentation
     * 
     * @param change_list 
     * @param patch_list 
     */
    void apply_change_list(std::vector<std::tuple<u64, i32, i32, i32>> change_list, SchedulerPatchList &patch_list);

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
    void split_patchdata(u64 key_orginal, Patch &p0, Patch &p1, Patch &p2, Patch &p3, Patch &p4, Patch &p5, Patch &p6,
                         Patch &p7);

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
    void merge_patchdata(u64 new_key, u64 old_key0, u64 old_key1, u64 old_key2, u64 old_key3, u64 old_key4, u64 old_key5,
                         u64 old_key6, u64 old_key7);


    inline SchedulerPatchData(PatchDataLayout & pdl) : pdl(pdl), sim_box(pdl){}
};