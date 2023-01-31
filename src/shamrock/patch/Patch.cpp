// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "Patch.hpp"
#include "shamsys/MpiDataTypeHandler.hpp"

MPI_Datatype patch_MPI_types_list[2];
int          patch_MPI_block_lens[2];
MPI_Aint     patch_MPI_offset[2];

Register_MPIDtypeInit(init_patch_type,"mpi patch type"){
    using namespace shamrock::patch;

    patch_MPI_block_lens[0] = 9; // 9 u64
    patch_MPI_block_lens[1] = 2; // 2 u32

    patch_MPI_types_list[0] = MPI_LONG;
    patch_MPI_types_list[1] = MPI_INT;

    patch_MPI_offset[0] = offsetof(shamrock::patch::Patch, id_patch);
    patch_MPI_offset[1] = offsetof(shamrock::patch::Patch, data_count);

    mpi::type_create_struct(2, patch_MPI_block_lens, patch_MPI_offset, patch_MPI_types_list, &patch_MPI_type);
    mpi::type_commit(&patch_MPI_type);

}

Register_MPIDtypeFree(free_patch_type,"mpi patch type"){
    using namespace shamrock::patch;

    mpi::type_free(&patch_MPI_type);

}