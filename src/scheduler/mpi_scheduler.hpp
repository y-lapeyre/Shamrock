#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <mpi.h>
#include <vector>

class MpiScheduler{public:

    std::vector<Patch> global_patch_list;
    std::vector<Patch> local_patch_list;

    inline void sync_cluster_patches(){
        mpi_handler::vector_allgatherv_ks(local_patch_list, patch_MPI_type, global_patch_list, patch_MPI_type, MPI_COMM_WORLD);
    }

};