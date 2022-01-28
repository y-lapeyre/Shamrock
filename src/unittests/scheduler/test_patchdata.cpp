#include "../shamrocktest.hpp"

#include <mpi.h>
#include <random>
#include <vector>

#include "../../scheduler/patchdata.hpp"
#include "../../scheduler/mpi_scheduler.hpp"

Test_start("patchdata::", sync_patchdata_layout, -1) {

    if (mpi_handler::world_rank == 0) {
        patchdata_layout::set(1, 8, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);

    Test_assert("sync nVarpos_s",patchdata_layout::nVarpos_s == 1);
    Test_assert("sync nVarpos_d",patchdata_layout::nVarpos_d == 8);
    Test_assert("sync nVarU1_s ",patchdata_layout::nVarU1_s  == 4);
    Test_assert("sync nVarU1_d ",patchdata_layout::nVarU1_d  == 6);
    Test_assert("sync nVarU3_s ",patchdata_layout::nVarU3_s  == 2);
    Test_assert("sync nVarU3_d ",patchdata_layout::nVarU3_d  == 1);

}