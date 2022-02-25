#pragma once

#include <cmath>
#include <map>

#include "patch.hpp"
#include "patchdata.hpp"

#include "scheduler_patch_list.hpp"

class SimulationBoxInfo{public:

    f32_3 min_box_sim_s;
    f32_3 max_box_sim_s;

    f64_3 min_box_sim_d;
    f64_3 max_box_sim_d;

    //TODO implement box size reduction here




    

    inline void reset_box_size(){

        if(patchdata_layout::nVarpos_s == 1) {
            min_box_sim_s = {HUGE_VAL_F32};
            max_box_sim_s = {- HUGE_VAL_F32};
        }

        if(patchdata_layout::nVarpos_d == 1) {
            min_box_sim_s = {HUGE_VAL_F64};
            max_box_sim_s = {- HUGE_VAL_F64};
        }

    }

};

class SchedulerPatchData{public:
    std::map<u64, PatchData> owned_data;

    SimulationBoxInfo sim_box;

    void apply_change_list(std::vector<std::tuple<u64, i32, i32,i32>> change_list,SchedulerPatchList& patch_list);



    void split_patchdata(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);
    void merge_patchdata(u64 new_key,u64 old_key0,u64 old_key1,u64 old_key2,u64 old_key3,u64 old_key4,u64 old_key5,u64 old_key6,u64 old_key7);
};