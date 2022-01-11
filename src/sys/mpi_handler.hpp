/**
 * @file mpi_handler.hpp
 * @author your name (you@domain.com)
 * @brief handle mpi routines
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once



#include <vector>
#define MPI_LOGGER_ENABLED

#include <cstdio>
#include "mpi_wrapper.hpp"
#include "../utils/string_utils.hpp"

#include "../aliases.hpp"

namespace mpi_handler{

    inline bool working = false;

    inline int world_rank, world_size;
    //inline Logger* global_logger;

    /**
    * @brief Get the processor name 
    * @return std::string processor name
    */
    std::string get_proc_name();

    /**
    * @brief initialize MPI comm and logger
    */
    void init();

    /**
    * @brief close MPI context
    * 
    */
    void close();





    



    /**
     * @brief allgatherv with knowing total count of object
     * 
     * @tparam T 
     * @param send_vec 
     * @param send_type 
     * @param recv_vec 
     * @param recv_type 
     */
    template<class T>
    inline void vector_allgatherv_ks(std::vector<u32> & send_vec ,MPI_Datatype send_type,std::vector<T> & recv_vec,MPI_Datatype recv_type,MPI_Comm comm){

        u32 local_count = send_vec.size();

        int* table_data_count = new int[world_size];

        //crash
        mpi::allgather(
            &local_count, 
            1, 
            MPI_INT, 
            &table_data_count[0], 
            1, 
            MPI_INT, 
            comm);

        //printf("table_data_count = [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);



        int* node_displacments_data_table = new int[world_size];

        node_displacments_data_table[0] = 0;

        for(u32 i = 1 ; i < world_size; i++){
            node_displacments_data_table[i] = node_displacments_data_table[i-1] + table_data_count[i-1];
        }
        
        //printf("node_displacments_data_table = [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);
        
        mpi::allgatherv(
            &send_vec[0], 
            send_vec.size(),
            send_type, 
            &recv_vec[0], 
            table_data_count, 
            node_displacments_data_table, 
            recv_type, 
            comm);


        delete [] table_data_count;
        delete [] node_displacments_data_table;
    }

}
