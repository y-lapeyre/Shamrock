// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"
#include "core/patch/patch_field.hpp"
#include "core/patch/base/patchdata_field.hpp"
#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "unittests/shamrocktest.hpp"
#include <random>



Test_start("patch/patchdata_field.cpp", patch_data_field_check_match,1){
    std::mt19937 eng(0x1111);  

    PatchDataField<f32> d_check("test",1);
    d_check.gen_mock_data(10000, eng);

    Test_assert("patch_data_check_match reflexivity", d_check.check_field_match(d_check));
}

Test_start("patch/patchdata_field.cpp", isend_irecv_f32, 2){
    std::mt19937 eng(0x1111);  

    

    create_sycl_mpi_types();



    PatchDataField<f32> d1_check("test",1);
    PatchDataField<f32> d2_check("test",1);

    std::uniform_int_distribution<u64> distu64(1,6000);

    d1_check.gen_mock_data(distu64(eng), eng);
    d2_check.gen_mock_data(distu64(eng), eng);

    std::vector<MPI_Request> rq_lst;
    PatchDataField<f32> recv_d("test",1);

    if(mpi_handler::world_rank == 0){
        patchdata_field::isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_field::irecv(recv_d,rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_field::isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_field::irecv(recv_d,rq_lst, 0, 0, MPI_COMM_WORLD);
    }

    std::cout << "request len : [" << mpi_handler::world_rank << "] " << rq_lst.size() << std::endl;

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());

    
    if(mpi_handler::world_rank == 0){
        PatchDataField<f32> & p1 = recv_d;
        PatchDataField<f32> & p2 = d2_check;
        Test_assert("recv_d == d1_check (f32)", p1.check_field_match(p2));
    }

    if(mpi_handler::world_rank == 1){
        PatchDataField<f32> & p1 = recv_d;
        PatchDataField<f32> & p2 = d1_check;
        Test_assert("recv_d == d1_check (f32)", p1.check_field_match(p2));
    }

    free_sycl_mpi_types();
}