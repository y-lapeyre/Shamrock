#pragma once

#include "../aliases.hpp"
#include "../sys/mpi_handler.hpp"

#ifdef aaa
/**
 * @brief Patch containing the information associated with each patch not the actual data 
 * the data in packed to reduce data transferd through MPI.
 */
struct Patch{

    //patch information (unique key)
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
    // FMM fields
    ///////////////////////////////////////////////




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
     * should_merge_child = flags & 0x1
     *       | if true during next patch tree update child (and the PatchData associated) of this patch will be merged
     *
     * should_split = flags & 0x2
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

#endif


struct Patch{

    //patch information (unique key)
    u64 id_patch;

    //load balancing fields
    u64 pack_node_index;
    u64 load_value;

    //Data
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;



    ///////////////////////////////////////////////
    // FMM fields
    ///////////////////////////////////////////////


    u32 data_count;
    
    u32 node_owner_id;

    




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



inline MPI_Datatype patch_MPI_type;
inline MPI_Datatype patch_MPI_types_list[2];
inline int          patch_MPI_block_lens[2];
inline MPI_Aint     patch_MPI_offset[2];

inline bool __mpi_patch_type_active = false;
inline bool is_mpi_patch_type_active(){
    return __mpi_patch_type_active;
}


void create_MPI_patch_type();

void free_MPI_patch_type();




