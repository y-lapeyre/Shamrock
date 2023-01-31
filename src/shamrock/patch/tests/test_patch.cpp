#include "shamrock/patch/Patch.hpp"

#include "shamrock/legacy/patch/base/patch.hpp"

#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/Patch.cpp:MpiType", patch_mpi_type, 2){

    using namespace shamrock::patch;

    Patch check_patch{};
    check_patch.id_patch = 156;
    check_patch.pack_node_index = 48414;
    check_patch.load_value = 4951956;
    check_patch.x_min = 0;
    check_patch.y_min = 1;
    check_patch.z_min = 2;
    check_patch.x_max = 3;
    check_patch.y_max = 8;
    check_patch.z_max = 6;
    check_patch.data_count = 7444444;
    check_patch.node_owner_id = 44444;



    if(shamsys::instance::world_rank == 0){
        mpi::send(&check_patch, 1, patch_MPI_type, 1, 0, MPI_COMM_WORLD);
    }

    if(shamsys::instance::world_rank == 1){
        Patch rpatch{};

        MPI_Status st;
        mpi::recv(&rpatch, 1, patch_MPI_type, 0, 0, MPI_COMM_WORLD, &st);

        shamtest::asserts().assert_bool("patch are equal", rpatch == check_patch);
    }
}