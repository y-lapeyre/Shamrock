// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patch.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file for the patch struct and related function 
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include "shamrock/patch/Patch.hpp"


#include "shamsys/legacy/mpi_handler.hpp"





/**
 * @brief patch related functions
 */
namespace patch {

    /**
     * @brief split patch \p p0 in p0 -> p7
     * 
     * @param p0 
     * @param p1 
     * @param p2 
     * @param p3 
     * @param p4 
     * @param p5 
     * @param p6 
     * @param p7 
     */
    void split_patch_obj(Patch & p0, Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);

    /**
     * @brief merge patch \p p0 -> p7 into p0
     * 
     * @param p0 
     * @param p1 
     * @param p2 
     * @param p3 
     * @param p4 
     * @param p5 
     * @param p6 
     * @param p7 
     */
    void merge_patch_obj(Patch & p0, Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7);


    /**
     * @brief the mpi patch type (ok if is_mpi_patch_type_active() return true)
     */
    inline MPI_Datatype patch_MPI_type;

    /**
     * @brief is mpi type active
     * 
     * @return true patch_MPI_type can be used   
     * @return false patch_MPI_type shouldnt be used
     */
    bool is_mpi_patch_type_active();


    /**
     * @brief Create the mpi type for the Patch struct
     */
    void create_MPI_patch_type();


    /**
     * @brief Destroy the mpi type for the Patch struct
     */
    void free_MPI_patch_type();

}
