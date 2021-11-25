#pragma once

#include "../aliases.hpp"
#include <mpi.h>

/**
 * @brief Patch containing the information associated with each patch not the actual data 
 * the data in packed to reduce data transferd through MPI.
 */
struct Patch{

    //patch information
    u64 id_patch;

    //patch tree info
    u64 id_parent;

    u64 id_child_r;
    u64 id_child_l;

    //Data
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;
    u32 data_count;

    ///////////////////////////////////////////////
    //patch tree evolution info
    ///////////////////////////////////////////////

    /**
     * @brief if true during next patch tree update child (and the PatchData associated) of this patch will be merged
     */
    bool should_merge_child;

    /**
     * @brief if true during next patch tree update this patch data will be split into two childs
     */
    bool should_split;

    inline bool operator==(const Patch& rhs){ 

        bool ret_val = true;

        ret_val = ret_val && (id_patch            ==rhs.id_patch          );
        ret_val = ret_val && (id_parent           ==rhs.id_parent         );
        ret_val = ret_val && (id_child_r          ==rhs.id_child_r        );
        ret_val = ret_val && (id_child_l          ==rhs.id_child_l        );
        ret_val = ret_val && (x_min               ==rhs.x_min             );
        ret_val = ret_val && (y_min               ==rhs.y_min             );
        ret_val = ret_val && (z_min               ==rhs.z_min             );
        ret_val = ret_val && (x_max               ==rhs.x_max             );
        ret_val = ret_val && (y_max               ==rhs.y_max             );
        ret_val = ret_val && (z_max               ==rhs.z_max             );
        ret_val = ret_val && (data_count          ==rhs.data_count        );
        ret_val = ret_val && (should_merge_child  ==rhs.should_merge_child);
        ret_val = ret_val && (should_split        ==rhs.should_split      );

        return ret_val;
    }

};

inline MPI_Datatype patch_MPI_type;
inline MPI_Datatype patch_MPI_types_list[3];
inline int          patch_MPI_block_lens[3];
inline MPI_Aint     patch_MPI_offset[3];

inline void create_MPI_patch_type(){
    
    patch_MPI_block_lens[0] = 10; // 10 u64
    patch_MPI_block_lens[1] = 1;  //  1 u32
    patch_MPI_block_lens[2] = 2;  //  2 bools

    patch_MPI_types_list[0] = MPI_LONG;
    patch_MPI_types_list[1] = MPI_INT;
    patch_MPI_types_list[2] = MPI_CXX_BOOL;

    //Patch a;

    // MPI_Get_address(&a.id_patch,            & patch_MPI_offset[0]);
    // MPI_Get_address(&a.id_parent,           & patch_MPI_offset[1]);
    // MPI_Get_address(&a.id_child_r,          & patch_MPI_offset[2]);
    // MPI_Get_address(&a.id_child_l,          & patch_MPI_offset[3]);
    // MPI_Get_address(&a.x_min,               & patch_MPI_offset[4]);
    // MPI_Get_address(&a.y_min,               & patch_MPI_offset[5]);
    // MPI_Get_address(&a.z_min,               & patch_MPI_offset[6]);
    // MPI_Get_address(&a.x_max,               & patch_MPI_offset[7]);
    // MPI_Get_address(&a.y_max,               & patch_MPI_offset[8]);
    // MPI_Get_address(&a.z_max,               & patch_MPI_offset[9]);
    // MPI_Get_address(&a.data_count,          & patch_MPI_offset[1]);
    // MPI_Get_address(&a.should_merge_child,  & patch_MPI_offset[2]);
    // MPI_Get_address(&a.should_split,        & patch_MPI_offset[12]);
    
    patch_MPI_offset[0] = offsetof(Patch, id_patch); 
    patch_MPI_offset[1] = offsetof(Patch, data_count);
    patch_MPI_offset[2] = offsetof(Patch, should_merge_child);
    

    MPI_Type_create_struct( 3, patch_MPI_block_lens, patch_MPI_offset, patch_MPI_types_list, &patch_MPI_type );
    MPI_Type_commit( &patch_MPI_type );
}

inline void free_MPI_patch_type(){
    MPI_Type_free(&patch_MPI_type);
}