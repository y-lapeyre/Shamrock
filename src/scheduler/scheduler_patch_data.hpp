#pragma once

#include <map>

#include "patch.hpp"
#include "patchdata.hpp"

#include "scheduler_patch_list.hpp"

class SchedulerPatchData{public:
    std::map<u64, PatchData> owned_data;

    void apply_change_list(std::vector<std::tuple<u64, i32, i32,i32>> change_list,SchedulerPatchList& patch_list);

    void split_patchdata(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);

    private:
    void split_patchdata_pos_s(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);
    void split_patchdata_pos_d(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);
};