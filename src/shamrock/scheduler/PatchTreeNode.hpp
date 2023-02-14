// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamrock/patch/PatchCoord.hpp"


namespace shamrock::scheduler {

    /**
     * @brief Node information in the PatchTree link list
     * 
     */
    class LinkedTreeNode{public:
        u32 level;
        u64 parent_nid;
        u64 childs_nid[8] {u64_max};

        bool is_leaf = true;
        bool child_are_all_leafs = false;
    };

    /**
     * @brief Node information in the patchtree + held patch info
     * 
     */
    class PatchTreeNode{public:
        patch::PatchCoord patch_coord;

        LinkedTreeNode tree_node;
        u64 linked_patchid;

        //patch fields
        u64 data_count = u64_max;
        u64 load_value = u64_max;

    };

    



}