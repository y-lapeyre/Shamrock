/**
 * @file patch.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of patch.hpp related functions
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "patch.hpp"

namespace patch {

    MPI_Datatype patch_MPI_types_list[2];
    int          patch_MPI_block_lens[2];
    MPI_Aint     patch_MPI_offset[2];
    bool __mpi_patch_type_active = false;

    void create_MPI_patch_type() {

        patch_MPI_block_lens[0] = 9; // 9 u64
        patch_MPI_block_lens[1] = 2; // 2 u32

        patch_MPI_types_list[0] = MPI_LONG;
        patch_MPI_types_list[1] = MPI_INT;

        patch_MPI_offset[0] = offsetof(Patch, id_patch);
        patch_MPI_offset[1] = offsetof(Patch, data_count);

        mpi::type_create_struct(2, patch_MPI_block_lens, patch_MPI_offset, patch_MPI_types_list, &patch_MPI_type);
        mpi::type_commit(&patch_MPI_type);

        __mpi_patch_type_active = true;
    }

    void free_MPI_patch_type() {
        mpi::type_free(&patch_MPI_type);

        __mpi_patch_type_active = false;
    }

    bool is_mpi_patch_type_active(){
        return __mpi_patch_type_active;
    }




    void split_patch_obj(
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

        p0.data_count /= 8;
        p0.load_value /= 8;

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





    void merge_patch_obj(
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

        u64 max_x = p0.x_max;
        u64 max_y = p0.y_max;
        u64 max_z = p0.z_max;

        min_x = sycl::min(min_x,p1.x_min);
        min_y = sycl::min(min_y,p1.y_min);
        min_z = sycl::min(min_z,p1.z_min);
        max_x = sycl::max(max_x,p1.x_max);
        max_y = sycl::max(max_y,p1.y_max);
        max_z = sycl::max(max_z,p1.z_max);

        min_x = sycl::min(min_x,p2.x_min);
        min_y = sycl::min(min_y,p2.y_min);
        min_z = sycl::min(min_z,p2.z_min);
        max_x = sycl::max(max_x,p2.x_max);
        max_y = sycl::max(max_y,p2.y_max);
        max_z = sycl::max(max_z,p2.z_max);

        min_x = sycl::min(min_x,p3.x_min);
        min_y = sycl::min(min_y,p3.y_min);
        min_z = sycl::min(min_z,p3.z_min);
        max_x = sycl::max(max_x,p3.x_max);
        max_y = sycl::max(max_y,p3.y_max);
        max_z = sycl::max(max_z,p3.z_max);

        min_x = sycl::min(min_x,p4.x_min);
        min_y = sycl::min(min_y,p4.y_min);
        min_z = sycl::min(min_z,p4.z_min);
        max_x = sycl::max(max_x,p4.x_max);
        max_y = sycl::max(max_y,p4.y_max);
        max_z = sycl::max(max_z,p4.z_max);

        min_x = sycl::min(min_x,p5.x_min);
        min_y = sycl::min(min_y,p5.y_min);
        min_z = sycl::min(min_z,p5.z_min);
        max_x = sycl::max(max_x,p5.x_max);
        max_y = sycl::max(max_y,p5.y_max);
        max_z = sycl::max(max_z,p5.z_max);

        min_x = sycl::min(min_x,p6.x_min);
        min_y = sycl::min(min_y,p6.y_min);
        min_z = sycl::min(min_z,p6.z_min);
        max_x = sycl::max(max_x,p6.x_max);
        max_y = sycl::max(max_y,p6.y_max);
        max_z = sycl::max(max_z,p6.z_max);

        min_x = sycl::min(min_x,p7.x_min);
        min_y = sycl::min(min_y,p7.y_min);
        min_z = sycl::min(min_z,p7.z_min);
        max_x = sycl::max(max_x,p7.x_max);
        max_y = sycl::max(max_y,p7.y_max);
        max_z = sycl::max(max_z,p7.z_max);

        p0.x_min = min_x;
        p0.y_min = min_y;
        p0.z_min = min_z;
        p0.x_max = max_x;
        p0.y_max = max_y;
        p0.z_max = max_z;

        p0.pack_node_index = u64_max;
        

    }

}