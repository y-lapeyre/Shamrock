/**
 * @file mpi_handler.cpp
 * @author your name (you@domain.com)
 * @brief implementation of mpi_handler.hpp
 * 
 * @copyright Copyright (c) 2021
 * 
 */



#include "mpi_handler.hpp"


std::string get_proc_name(){
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    return std::string(processor_name);
}

void mpi_barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}


void mpi_init(){
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    
    // unsigned int thread_cnt = 0;
    // #pragma omp parallel
    // {
    //     thread_cnt = omp_get_num_threads();
    // }
    
    
    //log_string_rank = format("[%03d]: ",world_rank);
    
    printf("[%03d]: \x1B[32mMPI_Init : node n°%03d | world size : %d | name = %s\033[0m\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    global_logger = new Logger(format("log_full_%04d",world_rank));


    global_logger->log("[%03d]: MPI_Init : node n°%03d | world size : %d | name = %s\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    mpi_barrier();
    if(world_rank == 0){
        printf("------------ MPI init ok ------------ \n");
    }

    global_logger->log("------------ MPI init ok ------------ \n");
    
}


void mpi_close(){    
    
    global_logger->log("------------ MPI_Finalize ------------\n");
    MPI_Finalize();   

    global_logger->log("deleting logger\n");
    
    delete global_logger;
}