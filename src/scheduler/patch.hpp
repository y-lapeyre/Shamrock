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

#include "../aliases.hpp"
#include "../sys/mpi_handler.hpp"



/**
 * @brief Patch object that contain generic patch information
 * 
 */
struct Patch{



    u64 id_patch; //unique key that identify the patch


    //load balancing fields

    u64 pack_node_index; ///< this value mean "to pack with index xxx in the global patch table" and not "to pack with id_pach == xxx"
    u64 load_value; ///< if synchronized contain the load value of the patch

    //Data
    u64 x_min; ///< box coordinate of the corresponding patch
    u64 y_min; ///< box coordinate of the corresponding patch
    u64 z_min; ///< box coordinate of the corresponding patch
    u64 x_max; ///< box coordinate of the corresponding patch
    u64 y_max; ///< box coordinate of the corresponding patch
    u64 z_max; ///< box coordinate of the corresponding patch


    u32 data_count; ///< number of element in the corresponding patchdata
    
    u32 node_owner_id;  ///< node rank owner of this patch

    



    /**
     * @brief check if patch equals
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    inline bool operator==(const Patch& rhs){ 

        bool ret_val = true;

        ret_val = ret_val && (id_patch            ==rhs.id_patch          );

        ret_val = ret_val && (pack_node_index     ==rhs.pack_node_index   );
        ret_val = ret_val && (load_value          ==rhs.load_value        );

        ret_val = ret_val && (x_min               ==rhs.x_min             );
        ret_val = ret_val && (y_min               ==rhs.y_min             );
        ret_val = ret_val && (z_min               ==rhs.z_min             );
        ret_val = ret_val && (x_max               ==rhs.x_max             );
        ret_val = ret_val && (y_max               ==rhs.y_max             );
        ret_val = ret_val && (z_max               ==rhs.z_max             );
        ret_val = ret_val && (data_count          ==rhs.data_count        );

        ret_val = ret_val && (node_owner_id       ==rhs.node_owner_id     );

        return ret_val;
    }

};


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
