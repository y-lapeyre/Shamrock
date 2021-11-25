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
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include "../io/logger.hpp"
#include "../utils/string_utils.hpp"

inline bool mpi_working = false;
inline int world_rank, world_size;
inline Logger* global_logger;

/**
 * @brief Get the processor name 
 * @return std::string processor name
 */
std::string get_proc_name();

/**
 * @brief call MPI_Barrier(MPI_COMM_WORLD);
 */
void mpi_barrier();

/**
 * @brief initialize MPI comm and logger
 */
void mpi_init();

/**
 * @brief close MPI context
 * 
 */
void mpi_close();
