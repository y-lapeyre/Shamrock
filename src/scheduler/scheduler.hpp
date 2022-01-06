#pragma once

#include "patch.hpp"

#include "../sys/mpi_handler.hpp"
#include <map>
#include <vector>

#include "patchdata.hpp"





namespace scheduler {

    /////////////
    //behavior flags
    /////////////

    /**
     * @brief is set to true if the scheduler should work with multiple nodes
     */
    inline bool mpi_mode = false;

    /**
     * @brief is set to true only if the scheduler is running
     */
    inline bool initialized = false;

    
    /**
     * @brief if true sync_patch_tree() will query the total patch count before rebuilding
     */
    inline bool should_query_total_patch_count = true;







    /////////////
    // data handler
    /////////////

    //shared by all MPI node
    inline std::vector<Patch> patch_table;

    inline std::vector<Patch> patch_table_local;
    //inline std::vector<u64>   owned_patches;



    inline std::map<u64, PatchData> owned_patch_data;


    




    /**
     * @brief init the mpi scheduler 
     * 
     */
    inline void init(){

        mpi_mode = false;

        if(mpi_working && world_size > 1){
            mpi_mode = true;
        }

        initialized = true;
    }

    inline void finalize(){
        mpi_mode = false;
        initialized = false;
    }

    /**
     * @brief sync patch tree between each node 
     * (aka) patch_table = gather of all patch_table_local of each node
     * the size of the vector patch_table will be update if should_query_total_patch_count == true
     */
    void sync_patch_tree();

    void select_owned_patches();
    







    u32 get_patch_count_from_local();

    /**
    * @brief WARNING unsafe function check that patch_table size is corect otherwise SEGFAULT
    * 
    */
    void rebuild_global_patch_table_from_local();

    void rebuild_local_patch_table_from_global();


    inline void sort_patch_list(std::vector<Patch>& patch_table,std::vector<u64>& keys){

        u32 lenght_table = patch_table.size();

        u32 i = 1;
        while(i < lenght_table){
            u32 j = i;
            while(j > 0 and keys[j-1] > keys[j]){
                std::swap(keys[j]       ,keys[j-1]);
                std::swap(patch_table[j],patch_table[j-1]);
                j = j - 1;
            }
            i ++;
        }

    }

    inline void balance_patch_load(std::vector<Patch>& patch_table, u32 world_size){
        
        u64 dt_count_sum = 0;

        for(Patch p : patch_table){
            dt_count_sum += p.data_count;
        }

        //TODO [potential issue] here must check that the conversion to double doesn't mess up the target dt_cnt or find another way
        double target_datacnt = double(dt_count_sum)/world_size;

        u64 current_dtcnt = 0;
        u64 current_node  = 0;

        for(u32 i = 0; i < patch_table.size(); i++){
            
            //TODO [to add] register list of operation done somehow maybe list of old node_owner_id
            patch_table[i].node_owner_id = current_node;

            current_dtcnt += patch_table[i].data_count;

            if(current_dtcnt > (current_node+1)*target_datacnt){
                current_node += 1;
            }

        }
    }

}

