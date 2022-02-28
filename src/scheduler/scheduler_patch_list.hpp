/**
 * @file scheduler_patch_list.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Class to handle the patch list of the mpi scheduler
 * @version 0.1
 * @date 2022-02-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "patch.hpp"

/**
 * @brief Handle the patch list of the mpi scheduler
 * 
 */
class SchedulerPatchList{public:

    //TODO move variable to private
    u64 _next_patch_id = 0;

    /**
     * @brief contain the list of all patches in the simulation
     */
    std::vector<Patch> global;

    /**
     * @brief contain the list of patch owned by the current node
     */
    std::vector<Patch> local;



    /**
     * @brief rebuild global from the local list of each tables
     *  
     * similar to \p global = allgather(\p local)
     */
    void sync_global();


    /**
     * @brief select owned patches owned by the node to rebuild local
     * 
     * @return std::unordered_set<u64> 
     */
    [[nodiscard]]
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
    void build_global_idx_map();


    std::unordered_map<u64,u64> id_patch_to_local_idx;
    void build_local_idx_map();

    void reset_local_pack_index();


    std::tuple<u64,u64,u64,u64,u64,u64,u64,u64> split_patch(u64 id_patch);
    void merge_patch( u64 idx0, u64 idx1, u64 idx2, u64 idx3, u64 idx4, u64 idx5, u64 idx6, u64 idx7);

    

};



std::vector<Patch> make_fake_patch_list(u32 total_dtcnt,u64 div_limit);