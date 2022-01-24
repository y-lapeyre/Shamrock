#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct PatchData{
    u64 a;
};


class SchedulerPatchList{public:

    std::vector<Patch> global;

    std::vector<Patch> local;

    inline void sync_global(){
        mpi_handler::vector_allgatherv(local, patch_MPI_type, global, patch_MPI_type, MPI_COMM_WORLD);
    }

    inline void build_local(){
        local.clear();
        for(const Patch &p : global){
            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);
            }
        }
    }

    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    inline void build_local_get_diff(
        const std::unordered_set<u64> & old_patchids, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx,
        std::unordered_set<u64> & new_patchids
        ){
        local.clear();

        for (u64 i = 0; i < global.size(); i++) {
            const Patch & p = global[i];

            bool was_owned = (old_patchids.find(p.id_patch) != old_patchids.end());

            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);
                new_patchids.insert(p.id_patch);

                if(!was_owned){
                    to_recv_idx.push_back(i);
                }
            }else{
                if(was_owned){
                    to_send_idx.push_back(i);
                }
            }
        }

    }


    /*
    inline void build_local_from_ids(const std::unordered_set<u64> & id_lst){
        local.clear();
        for(const Patch &p : global){
            if(id_lst.find(p.id_patch) != id_lst.end()){
                local.push_back(p);
            }
        }
    }
    */

};

class SchedulerPatchData {public: 

};

class MpiScheduler{public:


    SchedulerPatchList patch_list;


    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id;
    std::map<u64, PatchData> owned_patchdata;

    



    inline void correct_patch_ownership(){
        
    }


    inline MpiScheduler(){

    }

    inline virtual ~MpiScheduler(){

    }

};