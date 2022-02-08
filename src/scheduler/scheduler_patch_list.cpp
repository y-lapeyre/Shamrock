#include "scheduler_patch_list.hpp"
#include <vector>


void SchedulerPatchList::sync_global(){
    mpi_handler::vector_allgatherv(local, patch_MPI_type, global, patch_MPI_type, MPI_COMM_WORLD);   
}


std::unordered_set<u64> SchedulerPatchList::build_local(){

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

void SchedulerPatchList::build_local_differantial(std::unordered_set<u64> &patch_id_lst, std::vector<u64> &to_send_idx, std::vector<u64> &to_recv_idx){
    
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
