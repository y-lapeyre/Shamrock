/**
 * @file mpi_handler.hpp
 * @author your name (you@domain.com)
 * @brief handle mpi routines
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#include <cstdio>
#include "mpi_wrapper.hpp"
#include "../utils/string_utils.hpp"

namespace mpi{

    inline bool working = false;

    inline int world_rank, world_size;
    //inline Logger* global_logger;

    /**
    * @brief Get the processor name 
    * @return std::string processor name
    */
    std::string get_proc_name();

    /**
    * @brief call MPI_Barrier(MPI_COMM_WORLD);
    */
    void barrier();

    /**
    * @brief initialize MPI comm and logger
    */
    void init();

    /**
    * @brief close MPI context
    * 
    */
    void close();





    





    inline void vector_allgatherv(){

    }

}
