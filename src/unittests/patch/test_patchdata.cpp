// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_layout.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
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
    

    //std::cout << d2_check.fields_f32.size() << std::endl;

    /*
    if(mpi_handler::world_rank == 0){
        patchdata_field_isend(d1_check.fields_f32[0], rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_field_irecv(recv_d.fields_f32[0],rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_field_isend(d2_check.fields_f32[0], rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_field_irecv(recv_d.fields_f32[0],rq_lst, 0, 0, MPI_COMM_WORLD);
    }
    */

    std::cout << "request len : [" << mpi_handler::world_rank << "] " << rq_lst.size() << std::endl;

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());

    /*
    if(mpi_handler::world_rank == 0){
        Test_assert("recv_d == d2_check", patch_data_check_match(recv_d, d2_check));
    }

    if(mpi_handler::world_rank == 1){
        Test_assert("recv_d == d1_check", patch_data_check_match(recv_d, d1_check));
    }
    */


    //////tmp/////
    if(mpi_handler::world_rank == 0){

        PatchData & p1 = recv_d;
        PatchData & p2 = d2_check;
        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32.size(); idx++){
            Test_assert("recv_d == d1_check (f32)", p1.fields_f32[idx].check_field_match(p2.fields_f32[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_2.size(); idx++){
            Test_assert("recv_d == d1_check (f32_2)", p1.fields_f32_2[idx].check_field_match(p2.fields_f32_2[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_3.size(); idx++){
            Test_assert("recv_d == d1_check (f32_3)", p1.fields_f32_3[idx].check_field_match(p2.fields_f32_3[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_4.size(); idx++){
            Test_assert("recv_d == d1_check (f32_4)", p1.fields_f32_4[idx].check_field_match(p2.fields_f32_4[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_8.size(); idx++){
            Test_assert("recv_d == d1_check (f32_8)", p1.fields_f32_8[idx].check_field_match(p2.fields_f32_8[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_16.size(); idx++){
            Test_assert("recv_d == d1_check (f32_16)", p1.fields_f32_16[idx].check_field_match(p2.fields_f32_16[idx]));
        }



        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64.size(); idx++){
            Test_assert("recv_d == d1_check (f64)", p1.fields_f64[idx].check_field_match(p2.fields_f64[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_2.size(); idx++){
            Test_assert("recv_d == d1_check (f64_2)", p1.fields_f64_2[idx].check_field_match(p2.fields_f64_2[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_3.size(); idx++){
            Test_assert("recv_d == d1_check (f64_3)", p1.fields_f64_3[idx].check_field_match(p2.fields_f64_3[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_4.size(); idx++){
            Test_assert("recv_d == d1_check (f64_4)", p1.fields_f64_4[idx].check_field_match(p2.fields_f64_4[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_8.size(); idx++){
            Test_assert("recv_d == d1_check (f64_8)", p1.fields_f64_8[idx].check_field_match(p2.fields_f64_8[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_16.size(); idx++){
            Test_assert("recv_d == d1_check (f64_16)", p1.fields_f64_16[idx].check_field_match(p2.fields_f64_16[idx]));
        }



        for(u32 idx = 0; idx < p1.patchdata_layout.fields_u32.size(); idx++){
            Test_assert("recv_d == d1_check (u32)", p1.fields_u32[idx].check_field_match(p2.fields_u32[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_u64.size(); idx++){
            Test_assert("recv_d == d1_check (u64)", p1.fields_u64[idx].check_field_match(p2.fields_u64[idx]));
        }
    }

    if(mpi_handler::world_rank == 1){

        PatchData & p1 = recv_d;
        PatchData & p2 = d1_check;
        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32.size(); idx++){
            Test_assert("recv_d == d1_check (f32)", p1.fields_f32[idx].check_field_match(p2.fields_f32[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_2.size(); idx++){
            Test_assert("recv_d == d1_check (f32_2)", p1.fields_f32_2[idx].check_field_match(p2.fields_f32_2[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_3.size(); idx++){
            Test_assert("recv_d == d1_check (f32_3)", p1.fields_f32_3[idx].check_field_match(p2.fields_f32_3[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_4.size(); idx++){
            Test_assert("recv_d == d1_check (f32_4)", p1.fields_f32_4[idx].check_field_match(p2.fields_f32_4[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_8.size(); idx++){
            Test_assert("recv_d == d1_check (f32_8)", p1.fields_f32_8[idx].check_field_match(p2.fields_f32_8[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f32_16.size(); idx++){
            Test_assert("recv_d == d1_check (f32_16)", p1.fields_f32_16[idx].check_field_match(p2.fields_f32_16[idx]));
        }



        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64.size(); idx++){
            Test_assert("recv_d == d1_check (f64)", p1.fields_f64[idx].check_field_match(p2.fields_f64[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_2.size(); idx++){
            Test_assert("recv_d == d1_check (f64_2)", p1.fields_f64_2[idx].check_field_match(p2.fields_f64_2[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_3.size(); idx++){
            Test_assert("recv_d == d1_check (f64_3)", p1.fields_f64_3[idx].check_field_match(p2.fields_f64_3[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_4.size(); idx++){
            Test_assert("recv_d == d1_check (f64_4)", p1.fields_f64_4[idx].check_field_match(p2.fields_f64_4[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_8.size(); idx++){
            Test_assert("recv_d == d1_check (f64_8)", p1.fields_f64_8[idx].check_field_match(p2.fields_f64_8[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_f64_16.size(); idx++){
            Test_assert("recv_d == d1_check (f64_16)", p1.fields_f64_16[idx].check_field_match(p2.fields_f64_16[idx]));
        }



        for(u32 idx = 0; idx < p1.patchdata_layout.fields_u32.size(); idx++){
            Test_assert("recv_d == d1_check (u32)", p1.fields_u32[idx].check_field_match(p2.fields_u32[idx]));
        }

        for(u32 idx = 0; idx < p1.patchdata_layout.fields_u64.size(); idx++){
            Test_assert("recv_d == d1_check (u64)", p1.fields_u64[idx].check_field_match(p2.fields_u64[idx]));
        }
    }
    //////////////

    free_sycl_mpi_types();
}