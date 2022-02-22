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



inline void split_patch_obj(
        Patch & p0, 
        Patch & p1,
        Patch & p2,
        Patch & p3,
        Patch & p4,
        Patch & p5,
        Patch & p6,
        Patch & p7
    ){

    u64 min_x = p0.x_min;
    u64 min_y = p0.y_min;
    u64 min_z = p0.z_min;

    u64 split_x = (((p0.x_max - p0.x_min) + 1)/2) - 1 + min_x;
    u64 split_y = (((p0.y_max - p0.y_min) + 1)/2) - 1 + min_y;
    u64 split_z = (((p0.z_max - p0.z_min) + 1)/2) - 1 + min_z;

    u64 max_x = p0.x_max;
    u64 max_y = p0.y_max;
    u64 max_z = p0.z_max;

    p1 = p0;
    p2 = p0;
    p3 = p0;
    p4 = p0;
    p5 = p0;
    p6 = p0;
    p7 = p0;

    p0.x_min = min_x;
    p0.y_min = min_y;
    p0.z_min = min_z;
    p0.x_max = split_x;
    p0.y_max = split_y;
    p0.z_max = split_z;

    p1.x_min = min_x;
    p1.y_min = min_y;
    p1.z_min = split_z + 1;
    p1.x_max = split_x;
    p1.y_max = split_y;
    p1.z_max = max_z;

    p2.x_min = min_x;
    p2.y_min = split_y+1;
    p2.z_min = min_z;
    p2.x_max = split_x;
    p2.y_max = max_y;
    p2.z_max = split_z;  

    p3.x_min = min_x;
    p3.y_min = split_y+1;
    p3.z_min = split_z+1;
    p3.x_max = split_x;
    p3.y_max = max_y;
    p3.z_max = max_z;

    p4.x_min = split_x+1;
    p4.y_min = min_y;
    p4.z_min = min_z;
    p4.x_max = max_x;
    p4.y_max = split_y;
    p4.z_max = split_z;

    p5.x_min = split_x+1;
    p5.y_min = min_y;
    p5.z_min = split_z+1;
    p5.x_max = max_x;
    p5.y_max = split_y;
    p5.z_max = max_z;

    p6.x_min = split_x+1;
    p6.y_min = split_y+1;
    p6.z_min = min_z;
    p6.x_max = max_x;
    p6.y_max = max_y;
    p6.z_max = split_z;

    p7.x_min = split_x+1;
    p7.y_min = split_y+1;
    p7.z_min = split_z+1;
    p7.x_max = max_x;
    p7.y_max = max_y;
    p7.z_max = max_z;
        

}
