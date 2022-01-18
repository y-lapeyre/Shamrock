#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <unordered_map>
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




    
    inline void build_local_patch_list(){
        
    }

    inline MpiScheduler(){

    }

    inline virtual ~MpiScheduler(){

    }

};