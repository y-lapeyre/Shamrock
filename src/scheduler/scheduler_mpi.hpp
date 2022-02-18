#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "patchdata.hpp"

#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"



class SchedulerMPI{public:


    SchedulerPatchList patch_list;
    SchedulerPatchData patch_data;

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id;

    


    void sync_build_LB(bool global_patch_sync, bool balance_load);

    



    inline void init_mpi_required_types(){
        if(!is_mpi_sycl_interop_active()){
            create_sycl_mpi_types();
        }

        if(!is_mpi_patch_type_active()){
            create_MPI_patch_type();
        }
    }

    inline void free_mpi_required_types(){
        if(is_mpi_sycl_interop_active()){
            free_sycl_mpi_types();
        }

        if(is_mpi_patch_type_active()){
            free_MPI_patch_type();
        }
    }

    inline SchedulerMPI(){

    }

    inline virtual ~SchedulerMPI(){

    }

};