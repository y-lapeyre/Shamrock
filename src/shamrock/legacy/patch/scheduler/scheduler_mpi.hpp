// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

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
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
//#include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/legacy/patch/base/patchtree.hpp"
#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

/**
 * @brief The MPI scheduler
 * 
 */
class PatchScheduler{public:

    shamrock::patch::PatchDataLayout & pdl;

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

    PatchScheduler(shamrock::patch::PatchDataLayout & pdl, u64 crit_split,u64 crit_merge);

    ~PatchScheduler();



    std::string dump_status();


    inline void update_local_dtcnt_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].data_count = patch_data.owned_data.at(id).get_obj_cnt();
        }
    }

    inline void update_local_load_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].load_value = patch_data.owned_data.at(id).get_obj_cnt();
        }
    }


    template<class vectype>
    std::tuple<vectype,vectype> get_box_tranform();

    template<class vectype>
    std::tuple<vectype,vectype> get_box_volume();

    bool should_resize_box(bool node_in);

    template<class vectype>
    void set_box_volume(std::tuple<vectype,vectype> box);

    [[deprecated]]
    void dump_local_patches(std::string filename);

    std::vector<std::unique_ptr<shamrock::patch::PatchData>> gather_data(u32 rank);


    /**
     * @brief add patch to the scheduler
     *
     * //TODO find a better way to do this it is too error prone
     * 
     * @param p 
     * @param pdat 
     */
    //[[deprecated]]
    inline void add_patch(shamrock::patch::Patch & p, shamrock::patch::PatchData & pdat){
        p.id_patch = patch_list._next_patch_id;
        patch_list._next_patch_id ++;

        patch_list.global.push_back(p);

        patch_data.owned_data.insert({p.id_patch , pdat});

    }

    [[deprecated]]
    void sync_build_LB(bool global_patch_sync, bool balance_load);



    //template<class Function>
    //[[deprecated]]
    //inline void for_each_patch_buf(Function && fct){
//
    //    
//
    //    for (auto &[id, pdat] : patch_data.owned_data) {
//
    //        if (! pdat.is_empty()) {
//
//
    //            Patch &cur_p = patch_list.global[patch_list.id_patch_to_global_idx[id]];
//
    //            PatchDataBuffer pdatbuf = attach_to_patchData(pdat);
//
    //            //TODO should feed the sycl queue to the lambda
//
    //            fct(id,cur_p,pdatbuf);
    //        }
    //    }
//
    //}

    template<class Function>
    inline void for_each_patch_data(Function && fct){

        for (auto &[id, pdat] : patch_data.owned_data) {

            if (! pdat.is_empty()) {

                shamrock::patch::Patch &cur_p = patch_list.global[patch_list.id_patch_to_global_idx[id]];

                if(!cur_p.is_err_mode()){
                    fct(id,cur_p,pdat);
                }

            }
        }

    }

    template<class Function>
    inline void for_each_patch(Function && fct){

        

        for (auto &[id, pdat] : patch_data.owned_data) {

            if (! pdat.is_empty()) {


                shamrock::patch::Patch &cur_p = patch_list.global[patch_list.id_patch_to_global_idx[id]];


                //TODO should feed the sycl queue to the lambda
                if(!cur_p.is_err_mode()){
                    fct(id,cur_p);
                }
            }
        }

    }

    //template<class Function, class Pfield>
    //inline void compute_patch_field(Pfield & field, MPI_Datatype & dtype , Function && lambda){
    //    field.local_nodes_value.resize(patch_list.local.size());
    //
    //    
    //
    //    for (u64 idx = 0; idx < patch_list.local.size(); idx++) {
    //
    //        Patch &cur_p = patch_list.local[idx];
    //
    //        PatchDataBuffer pdatbuf = attach_to_patchData(patch_data.owned_data.at(cur_p.id_patch));
    //
    //        field.local_nodes_value[idx] = lambda(shamsys::instance::get_compute_queue(),cur_p,pdatbuf);
    //
    //    }
    //
    //    field.build_global(dtype);
    //
    //}






    template<class Function, class Pfield>
    inline void compute_patch_field(Pfield & field, MPI_Datatype & dtype , Function && lambda){
        field.local_nodes_value.resize(patch_list.local.size());

        for (u64 idx = 0; idx < patch_list.local.size(); idx++) {

            shamrock::patch::Patch &cur_p = patch_list.local[idx];

            if(!cur_p.is_err_mode()){
                field.local_nodes_value[idx] = lambda(shamsys::instance::get_compute_queue(),cur_p,patch_data.owned_data.at(cur_p.id_patch));
            }
        }

        field.build_global(dtype);

    }



    private:


    
    void split_patches(std::unordered_set<u64> split_rq);
    void merge_patches(std::unordered_set<u64> merge_rq);

    void set_patch_pack_values(std::unordered_set<u64> merge_rq);

};