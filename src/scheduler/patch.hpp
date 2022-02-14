#pragma once

#include "../aliases.hpp"
#include "../sys/mpi_handler.hpp"




struct Patch{

    //patch information (unique key)
    u64 id_patch;

    //load balancing fields
    u64 pack_node_index; //be carefull this value mean "to pack with index xxx in the global patch table" and not "to pack with id_pach == xxx"
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




