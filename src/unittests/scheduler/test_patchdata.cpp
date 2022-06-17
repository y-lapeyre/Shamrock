// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "core/patch/base/patchdata_layout.hpp"
#include "unittests/shamrocktest.hpp"

#include <random>
#include <vector>


#include "core/patch/base/patchdata.hpp"
#include "core/patch/scheduler_mpi.hpp"

/*
Test_start("patchdata::", sync_patchdata_layout, -1) {

    if (mpi_handler::world_rank == 0) {
        patchdata_layout::set(1, 0, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);

    Test_assert("sync nVarpos_s",patchdata_layout::nVarpos_s == 1);
    Test_assert("sync nVarpos_d",patchdata_layout::nVarpos_d == 0);
    Test_assert("sync nVarU1_s ",patchdata_layout::nVarU1_s  == 4);
    Test_assert("sync nVarU1_d ",patchdata_layout::nVarU1_d  == 6);
    Test_assert("sync nVarU3_s ",patchdata_layout::nVarU3_s  == 2);
    Test_assert("sync nVarU3_d ",patchdata_layout::nVarU3_d  == 1);

}
*/


Test_start("patchdata::", send_recv_patchdata, 2){

    std::mt19937 eng(0x1111);  


    PatchDataLayout pdl;

    pdl.add_field<f32_3>("xyz", 1);
    pdl.xyz_mode = xyz32;

    pdl.add_field<f64_8>("test", 2);
    create_sycl_mpi_types();



    PatchData d1_check = patchdata_gen_dummy_data (pdl,eng);
    PatchData d2_check = patchdata_gen_dummy_data (pdl,eng);



    std::vector<MPI_Request> rq_lst;
    PatchData recv_d(pdl);

    if(mpi_handler::world_rank == 0){
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_irecv(recv_d,rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_irecv(recv_d,rq_lst, 0, 0, MPI_COMM_WORLD);
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
