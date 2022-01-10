/**
 * @file mpi_handler.cpp
 * @author your name (you@domain.com)
 * @brief implementation of mpi_handler.hpp
 * 
 * @copyright Copyright (c) 2021
 * 
 */



#include "mpi_handler.hpp"
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>


#define MPI_LOGGER_ENABLED



std::string rename_com(MPI_Comm m){
    std::stringstream ss;
    if(m == MPI_COMM_WORLD){
        ss << "MPI_COMM_WORLD";
    }else{
        ss << m;
    }
    return ss.str();
}



void mpi::init(){
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    working = true;
    
    //MPI errors will return error code

    int error ;
    error = MPI_Comm_set_errhandler (
        MPI_COMM_WORLD ,
        MPI_ERRORS_RETURN ) ;
    
    // unsigned int thread_cnt = 0;
    // #pragma omp parallel
    // {
    //     thread_cnt = omp_get_num_threads();
    // }
    
    
    //log_string_rank = format("[%03d]: ",world_rank);
    
    printf("[%03d]: \x1B[32mMPI_Init : node n°%03d | world size : %d | name = %s\033[0m\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    //global_logger = new Logger(format("log_full_%04d",world_rank));

    //global_logger->log("[%03d]: MPI_Init : node n°%03d | world size : %d | name = %s\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    barrier();
    //if(world_rank == 0){
    printf("------------ MPI init ok ------------ \n");
    //}

    //global_logger->log("------------ MPI init ok ------------ \n");
    
}


void mpi::close(){    
    
    //global_logger->log("------------ MPI_Finalize ------------\n");
    printf("------------ MPI_Finalize ------------\n");
    MPI_Finalize();   

    working = false;

    //global_logger->log("deleting logger\n");
    
    //delete global_logger;
}



void handle_errorcode(int errorcode){
    int length;
    char message[MPI_MAX_ERROR_STRING] ;

    MPI_Error_string ( errorcode , message , & length ) ;
    printf ("%.*s\n", length , message);

    mpi::abort(MPI_COMM_WORLD, 1);
}

void mpi::barrier(){
    mpi::barrier(MPI_COMM_WORLD);
}



std::string mpi::get_proc_name(){

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    int err_code = MPI_Get_processor_name(processor_name, &name_len);

    if(err_code != MPI_SUCCESS){
        handle_errorcode(err_code);
    }
    
    return std::string(processor_name);
}












