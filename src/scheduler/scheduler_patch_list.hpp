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

    

};