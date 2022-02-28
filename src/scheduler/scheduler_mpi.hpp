#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "patchdata.hpp"

#include "patchtree.hpp"
#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"



class SchedulerMPI{public:

    u64 crit_patch_split;
    u64 crit_patch_merge;


    SchedulerPatchList patch_list;
    SchedulerPatchData patch_data;
    PatchTree patch_tree;

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id;


    


    void sync_build_LB(bool global_patch_sync, bool balance_load);

    void scheduler_step(bool do_split_merge,bool do_load_balancing);
    
    inline void add_patch(Patch & p, PatchData & pdat){
        p.id_patch = patch_list._next_patch_id;
        patch_list._next_patch_id ++;

        patch_list.global.push_back(p);

        patch_data.owned_data[p.id_patch] = pdat;

        
    }

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

    inline SchedulerMPI(u64 crit_split,u64 crit_merge){

        crit_patch_split = crit_split;
        crit_patch_merge = crit_merge;
        
    }

    inline virtual ~SchedulerMPI(){

    }



    std::string dump_status();


    inline void update_local_dtcnt_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].data_count = patch_data.owned_data[id].pos_s.size() + patch_data.owned_data[id].pos_d.size() ;
        }
    }

    inline void update_local_load_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].load_value = patch_data.owned_data[id].pos_s.size() + patch_data.owned_data[id].pos_d.size() ;
        }
    }

    private:


    
    void split_patches(std::unordered_set<u64> split_rq);
    void merge_patches(std::unordered_set<u64> merge_rq);

    void set_patch_pack_values(std::unordered_set<u64> merge_rq);

   


};