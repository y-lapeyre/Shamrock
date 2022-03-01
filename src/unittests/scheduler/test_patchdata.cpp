#include "unittests/shamrocktest.hpp"

#include <mpi.h>
#include <random>
#include <vector>


#include "scheduler/patchdata.hpp"
#include "scheduler/scheduler_mpi.hpp"


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



Test_start("patchdata::", send_recv_patchdata, 2){

    std::mt19937 eng(0x1111);  


    if (mpi_handler::world_rank == 0) {
        patchdata_layout::set(1, 8, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);
    create_sycl_mpi_types();



    PatchData d1_check = patchdata_gen_dummy_data (eng);
    PatchData d2_check = patchdata_gen_dummy_data (eng);



    std::vector<MPI_Request> rq_lst;
    PatchData recv_d;

    if(mpi_handler::world_rank == 0){
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        recv_d = patchdata_irecv(rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        recv_d = patchdata_irecv(rq_lst, 0, 0, MPI_COMM_WORLD);
    }

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());


    if(mpi_handler::world_rank == 0){
        Test_assert("recv_d == d2_check", patch_data_check_match(recv_d, d2_check));
    }

    if(mpi_handler::world_rank == 1){
        Test_assert("recv_d == d1_check", patch_data_check_match(recv_d, d1_check));
    }


    free_sycl_mpi_types();
}