#include "patch.hpp"
#include <mpi.h>



void create_MPI_patch_type(){
    
    patch_MPI_block_lens[0] = 9; // 9 u64
    patch_MPI_block_lens[1] = 2; // 2 u32

    patch_MPI_types_list[0] = MPI_LONG;
    patch_MPI_types_list[1] = MPI_INT;
    
    patch_MPI_offset[0] = offsetof(Patch, id_patch); 
    patch_MPI_offset[1] = offsetof(Patch, data_count);

    mpi::type_create_struct( 2, patch_MPI_block_lens, patch_MPI_offset, patch_MPI_types_list, &patch_MPI_type );
    mpi::type_commit( &patch_MPI_type );

    __mpi_patch_type_active = true;
}

void free_MPI_patch_type(){
    mpi::type_free(&patch_MPI_type);

    __mpi_patch_type_active = false;
}