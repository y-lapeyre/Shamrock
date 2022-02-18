#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "patch.hpp"

class SchedulerPatchList{public:

    std::vector<Patch> global;
    std::vector<Patch> local;




    void sync_global();




    std::unordered_set<u64> build_local();

    
    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    void build_local_differantial(
        std::unordered_set<u64> & patch_id_lst, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx
        );




    std::unordered_map<u64,u64> id_patch_to_global_idx;
    inline void build_global_idx_map(){
        id_patch_to_global_idx.clear();

        u64 idx = 0;
        for(Patch p : global){
            id_patch_to_global_idx[p.id_patch]  = idx;
            idx ++;
        }

    }


    std::unordered_map<u64,u64> id_patch_to_local_idx;
    inline void build_local_idx_map(){
        id_patch_to_local_idx.clear();

        u64 idx = 0;
        for(Patch p : local){
            id_patch_to_local_idx[p.id_patch]  = idx;
            idx ++;
        }

    }
    

};



std::vector<Patch> make_fake_patch_list(u32 total_dtcnt,u64 div_limit);