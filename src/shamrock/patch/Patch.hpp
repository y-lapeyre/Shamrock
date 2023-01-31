// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Patch.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file for the patch struct and related function 
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once


#include "aliases.hpp"


namespace shamrock::patch {




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




} // namespace shamrock::patch