// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "unittests/shamrocktest.hpp"
#include <random>


Test_start("patch/patchdata.cpp", patch_data_check_match,1){
    std::mt19937 eng(0x1111);  


    PatchDataLayout pdl;
    pdl.xyz_mode = xyz32;
    
    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);

    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3'", 1);
    pdl.add_field<f32_3>("f32_3''", 1);

    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 1);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);

    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);

    PatchData d_check = patchdata_gen_dummy_data (pdl,eng);

    Test_assert("patch_data_check_match reflexivity", patch_data_check_match(d_check, d_check));
}

Test_start("patch/patchdata.cpp", isend_irecv, 2){
    std::mt19937 eng(0x1111);  

    create_sycl_mpi_types();

    PatchDataLayout pdl;
    pdl.xyz_mode = xyz32;

    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);

    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3'", 2);
    pdl.add_field<f32_3>("f32_3''", 1);

    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 1);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);

    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);

    



    PatchData d1_check = patchdata_gen_dummy_data (pdl,eng);
    PatchData d2_check = patchdata_gen_dummy_data (pdl,eng);



    std::vector<PatchDataMpiRequest> rq_lst;
    PatchData recv_d(pdl);

    
    if(mpi_handler::world_rank == 0){
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_irecv(recv_d,rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_irecv(recv_d,rq_lst, 0, 0, MPI_COMM_WORLD);
    }
    



    //std::cout << "request len : [" << mpi_handler::world_rank << "] " << rq_lst.size() << std::endl;

    waitall_pdat_mpi_rq(rq_lst);

    
    if(mpi_handler::world_rank == 0){
        Test_assert("recv_d == d2_check", patch_data_check_match(recv_d, d2_check));
    }

    if(mpi_handler::world_rank == 1){
        Test_assert("recv_d == d1_check", patch_data_check_match(recv_d, d1_check));
    }
    



    free_sycl_mpi_types();
}