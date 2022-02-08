#pragma once

#include <map>

#include "patch.hpp"
#include "patchdata.hpp"

#include "scheduler_patch_list.hpp"

class SchedulerPatchData{public:
    std::map<u64, PatchData> owned_data;

    void apply_change_list(std::vector<std::tuple<u64, i32, i32,i32>> change_list,SchedulerPatchList& patch_list);

};