#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "patchdata.hpp"

class SchedulerPatchList{public:

    std::vector<Patch> global;

    std::vector<Patch> local;

    inline void sync_global(){
        mpi_handler::vector_allgatherv(local, patch_MPI_type, global, patch_MPI_type, MPI_COMM_WORLD);
    }

    inline std::unordered_set<u64> build_local(){

        std::unordered_set<u64> out_ids;

        local.clear();
        for(const Patch &p : global){
            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);
                out_ids.insert(p.id_patch);
            }
        }

        return out_ids;
        
    }

    

    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    inline void build_local_differantial(
        std::unordered_set<u64> & patch_id_lst, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx
        ){
        local.clear();

        for (u64 i = 0; i < global.size(); i++) {
            const Patch & p = global[i];

            bool was_owned = (patch_id_lst.find(p.id_patch) != patch_id_lst.end());

            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);

                if(!was_owned){
                    to_recv_idx.push_back(i);
                    patch_id_lst.insert(p.id_patch);
                }
            }else{
                if(was_owned){
                    to_send_idx.push_back(i);
                    patch_id_lst.erase(p.id_patch);
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