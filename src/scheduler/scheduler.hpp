#pragma once

#include "patch.hpp"

#include "../sys/mpi_handler.hpp"
#include <map>

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




    inline void balance_patch_load(std::vector<Patch>& patch_table, u32 world_size){
        
    }

}

