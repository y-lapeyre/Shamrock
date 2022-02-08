#include "scheduler_mpi.hpp"

#include "loadbalancing.hpp"

void SchedulerMPI::sync_build_LB(bool global_sync, bool balance_load){

    if(global_sync) patch_list.sync_global();

    if(balance_load){
        //real load balancing
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = make_change_list(patch_list.global);

        //exchange data
        patch_data.apply_change_list(change_list, patch_list);
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}