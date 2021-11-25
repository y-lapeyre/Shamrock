#pragma once

#include "patch.hpp"

#include "../sys/mpi_handler.hpp"

namespace scheduler {
    /**
     * @brief is set to true if the scheduler should work with multiple nodes
     */
    inline bool mpi_mode = false;

    /**
     * @brief is set to true only if the scheduler is running
     */
    inline bool initialized = false;

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
}











//shared by all MPI node
inline std::vector<Patch> patch_table;

inline std::vector<Patch> patch_table_local;

inline u32 get_patch_count_from_local(){

    u32 local_len = patch_table_local.size();
    u32 global_len;
    MPI_Allreduce(&local_len, &global_len, 1, MPI_LONG , MPI_SUM, MPI_COMM_WORLD);
    return global_len;
}



/**
 * @brief WARNING unsafe function check that patch_table size is corect otherwise SEGFAULT
 * 
 */
inline void rebuild_global_patch_table_from_local(){


    u32 local_count = patch_table_local.size();

    int* table_patch_count = new int[world_size];

    //crash
    MPI_Allgather(
        &local_count, 
        1, 
        MPI_INT, 
        &table_patch_count[0], 
        1, 
        MPI_INT, 
        MPI_COMM_WORLD);

    printf("table_patch_count = [%d,%d,%d,%d]\n",table_patch_count[0],table_patch_count[1],table_patch_count[2],table_patch_count[3]);



    int* node_displacments_patch_table = new int[world_size];

    node_displacments_patch_table[0] = 0;

    for(u32 i = 1 ; i < world_size; i++){
        node_displacments_patch_table[i] = node_displacments_patch_table[i-1] + table_patch_count[i-1];
    }
    
    


    printf("node_displacments_patch_table = [%d,%d,%d,%d]\n",node_displacments_patch_table[0],node_displacments_patch_table[1],node_displacments_patch_table[2],node_displacments_patch_table[3]);



    MPI_Allgatherv(
        &patch_table_local[0], 
        patch_table_local.size(),
        patch_MPI_type, 
        &patch_table[0], 
        table_patch_count, 
        node_displacments_patch_table, 
        patch_MPI_type, 
        MPI_COMM_WORLD);




    delete [] table_patch_count;
    delete [] node_displacments_patch_table;
    

    printf("all gather par : %zu %zu\n", 
        patch_table_local.size(), 
        patch_table.size());

    
}