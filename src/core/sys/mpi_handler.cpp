// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file mpi_handler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of mpi_handler.hpp
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include "mpi_handler.hpp"
#include <iostream>
#include <sstream>
#include <string>




std::string rename_com(MPI_Comm m){
    std::stringstream ss;
    if(m == MPI_COMM_WORLD){
        ss << "MPI_COMM_WORLD";
    }else{
        ss << m;
    }
    return ss.str();
}



void mpi_handler::init(int argc, char *argv[]){

    #ifdef MPI_LOGGER_ENABLED
    std::cout << "%MPI_DEFINE:MPI_COMM_WORLD="<<MPI_COMM_WORLD<<"\n";
    #endif


    mpi::init(NULL, NULL);
    mpi::comm_size(MPI_COMM_WORLD, &world_size);
    mpi::comm_rank(MPI_COMM_WORLD, &world_rank);

    uworld_rank = world_rank;
    uworld_size = world_size;


    if(world_size < 1){
        throw "";
    }

    if(world_rank < 0){
        throw "";
    }


    #ifdef MPI_LOGGER_ENABLED
    std::cout << "%MPI_VALUE:world_size="<<world_size<<"\n";
    std::cout << "%MPI_VALUE:world_rank="<<world_rank<<"\n";
    #endif

    working = true;
    
    //MPI errors will return error code

    int error ;
    //error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    
    
    if(error != MPI_SUCCESS){
        throw "";
    }

    //log_string_rank = format("[%03d]: ",world_rank);
    
    printf("[%03d]: \x1B[32mMPI_Init : node n°%03d | world size : %d | name = %s\033[0m\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    //global_logger = new Logger(format("log_full_%04d",world_rank));

    //global_logger->log("[%03d]: MPI_Init : node n°%03d | world size : %d | name = %s\n",world_rank,world_rank,world_size,get_proc_name().c_str());

    mpi::barrier(MPI_COMM_WORLD);
    //if(world_rank == 0){
    printf("------------ MPI init ok ------------ \n");
    //}

    //global_logger->log("------------ MPI init ok ------------ \n");
    
}


void mpi_handler::close(){    
    
    //global_logger->log("------------ MPI_Finalize ------------\n");
    printf("------------ MPI_Finalize ------------\n");
    mpi::finalize();   

    working = false;

    //global_logger->log("deleting logger\n");
    
    //delete global_logger;
}



void handle_errorcode(int errorcode){
    int length;
    char message[MPI_MAX_ERROR_STRING] ;

    mpi::error_string ( errorcode , message , & length ) ;
    printf ("%.*s\n", length , message);

    mpi::abort(MPI_COMM_WORLD, 1);
}




std::string mpi_handler::get_proc_name(){

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    int err_code = mpi::get_processor_name(processor_name, &name_len);

    if(err_code != MPI_SUCCESS){
        handle_errorcode(err_code);
    }
    
    return std::string(processor_name);
}












