#pragma once

#include "../aliases.hpp"
#include "../sys/mpi_handler.hpp"


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
    // MPI
    ///////////////////////////////////////////////

    /**
     * @brief id of the node owning this patch
     */
    u32 node_owner_id;


    ///////////////////////////////////////////////
    //patch tree evolution info
    ///////////////////////////////////////////////

    /**
     * @brief flags to store behavior
     * should_merge_child = flags && 0x1
     *       | if true during next patch tree update child (and the PatchData associated) of this patch will be merged
     *
     * should_split = flags && 0x2
     *       | if true during next patch tree update this patch data will be split into two childs
     */
    u8 flags;



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

        ret_val = ret_val && (node_owner_id       ==rhs.node_owner_id     );

        ret_val = ret_val && (flags               ==rhs.flags             );

        return ret_val;
    }

};

inline MPI_Datatype patch_MPI_type;
inline MPI_Datatype patch_MPI_types_list[3];
inline int          patch_MPI_block_lens[3];
inline MPI_Aint     patch_MPI_offset[3];

void create_MPI_patch_type();

void free_MPI_patch_type();