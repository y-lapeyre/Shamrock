/**
 * @file scheduler_mpi.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief MPI scheduler
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <fstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchtree.hpp"
#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"
#include "sys/sycl_mpi_interop.hpp"

/**
 * @brief The MPI scheduler
 * 
 */
class SchedulerMPI{public:

    u64 crit_patch_split; ///< splitting limit (if load value > crit_patch_split => patch split)
    u64 crit_patch_merge; ///< merging limit (if load value < crit_patch_merge => patch merge)


    SchedulerPatchList patch_list; ///< handle the list of the patches of the scheduler
    SchedulerPatchData patch_data; ///< handle the data of the patches of the scheduler
    PatchTree patch_tree; ///< handle the tree structure of the patches

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id; ///< list of owned patch ids updated with (owned_patch_id = patch_list.build_local())


    


    /**
     * @brief scheduler step
     * 
     * @param do_split_merge 
     * @param do_load_balancing 
     */
    void scheduler_step(bool do_split_merge,bool do_load_balancing);
    
    
    void init_mpi_required_types();
    
    void free_mpi_required_types();

    SchedulerMPI(u64 crit_split,u64 crit_merge);

    ~SchedulerMPI();



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


    template<class vectype>
    std::tuple<vectype,vectype> get_box_tranform();

    inline bool should_resize_box(bool node_in){
        u16 tmp = node_in;
        u16 out = 0;
        mpi::allreduce(&tmp, &out, 1, mpi_type_u16, MPI_MAX, MPI_COMM_WORLD);
        return out;
    }

    [[deprecated]]
    void dump_local_patches(std::string filename);


    /**
     * @brief add patch to the scheduler
     *
     * //TODO find a better way to do this it is too error prone
     * 
     * @param p 
     * @param pdat 
     */
    [[deprecated]]
    inline void add_patch(Patch & p, PatchData & pdat){
        p.id_patch = patch_list._next_patch_id;
        patch_list._next_patch_id ++;

        patch_list.global.push_back(p);

        patch_data.owned_data[p.id_patch] = pdat;

        
    }

    [[deprecated]]
    void sync_build_LB(bool global_patch_sync, bool balance_load);


    private:


    
    void split_patches(std::unordered_set<u64> split_rq);
    void merge_patches(std::unordered_set<u64> merge_rq);

    void set_patch_pack_values(std::unordered_set<u64> merge_rq);

};