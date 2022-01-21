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

class MpiScheduler{public:

    std::vector<Patch> global_patch_list;
    std::vector<Patch> local_patch_list;

    inline void sync_cluster_patches(){
        mpi_handler::vector_allgatherv(local_patch_list, patch_MPI_type, global_patch_list, patch_MPI_type, MPI_COMM_WORLD);
    }



    std::unordered_set<u64> owned_patch_id;
    inline void build_local_patch_list(){
        local_patch_list.clear();
        for(const Patch &p : global_patch_list){
            if(owned_patch_id.find(p.id_patch) != owned_patch_id.end()){
                local_patch_list.push_back(p);
            }
        }
    }

    inline MpiScheduler(){

    }

    inline virtual ~MpiScheduler(){

    }

};