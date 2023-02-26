// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/random/random.hpp"
#include "shamtest/shamtest.hpp"
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <random>

#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"
#include "shamsys/SyclHelper.hpp"


template<class T> void test_constructor_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){


    u32 npart = 1e5;

    sycl::buffer<T> buf_comp = shamalgs::random::mock_buffer<T>(seed, npart);

    using namespace shamsys::comm;

    CommBuffer buf {buf_comp, prot};

    sycl::buffer<T> buf_comp2 = CommBuffer<sycl::buffer<T>>::convert(std::move(buf));

    shamtest::asserts().assert_equal(prefix+std::string("same size"), buf_comp.size(), buf_comp2.size());

    {
        sycl::host_accessor acc1 {buf_comp};
        sycl::host_accessor acc2 {buf_comp2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for(u32 i = 0; i < npart; i++){
            if(!shamsys::syclhelper::test_sycl_eq(acc1[i] , acc2[i])){
                eq = false;
                //id_err_list += std::to_string(i) + " ";
            }
        }

        if (eq) {
            shamtest::asserts().assert_bool("same content", eq);
        }else{
            shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
        }
    }
}

TestStart(Unittest, "shamsys/comm/comm-buffer/syclbuffer-constructor", constructordestructor, 1){

    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_constructor_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_constructor_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_constructor_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_constructor_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_constructor_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_constructor_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_constructor_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_constructor_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_constructor_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);
    
}





















template<class T> void test_comm_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    
    u32 npart = 1e4;

    sycl::buffer<T> buf_comp = shamalgs::random::mock_buffer<T>(seed, npart);

    using namespace shamsys::comm;



    if(shamsys::instance::world_rank == 0){

        CommBuffer buf {buf_comp, prot};


        CommRequests rqs;
        buf.isend(rqs, 1,0,MPI_COMM_WORLD);

        rqs.wait_all();

    }else if (shamsys::instance::world_rank == 1) {
        
        CommDetails<sycl::buffer<T>> details;

        details.comm_len = npart;

        CommBuffer buf {details,prot};

        CommRequests rqs;
        buf.irecv(rqs, 0,0,MPI_COMM_WORLD);
        rqs.wait_all();



        sycl::buffer<T> buf_comp2 = CommBuffer<sycl::buffer<T>>::convert(std::move(buf));

        shamtest::asserts().assert_equal(prefix+std::string("same size"), buf_comp.size(), buf_comp2.size());

        {
            sycl::host_accessor acc1 {buf_comp};
            sycl::host_accessor acc2 {buf_comp2};

            std::string id_err_list = "errors in id : ";

            bool eq = true;
            for(u32 i = 0; i < npart; i++){
                if(!shamsys::syclhelper::test_sycl_eq(acc1[i] , acc2[i])){
                    eq = false;
                    //id_err_list += std::to_string(i) + " ";
                }
            }

            if (eq) {
                shamtest::asserts().assert_bool("same content", eq);
            }else{
                shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
            }
        }
    }

}




template<class T> void test_comm_probe_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    
    u32 npart = 1e4;

    sycl::buffer<T> buf_comp = shamalgs::random::mock_buffer<T>(seed, npart);

    using namespace shamsys::comm;



    if(shamsys::instance::world_rank == 0){

        CommBuffer buf {buf_comp, prot};


        CommRequests rqs;
        buf.isend(rqs, 1,0,MPI_COMM_WORLD);

        rqs.wait_all();

    }else if (shamsys::instance::world_rank == 1) {
        

        CommRequests rqs;
        auto buf = CommBuffer<sycl::buffer<T>>::irecv_probe(rqs, 0,0,MPI_COMM_WORLD,prot,{});

        rqs.wait_all();



        sycl::buffer<T> buf_comp2 = CommBuffer<sycl::buffer<T>>::convert(std::move(buf));

        shamtest::asserts().assert_equal(prefix+std::string("same size"), buf_comp.size(), buf_comp2.size());

        {
            sycl::host_accessor acc1 {buf_comp};
            sycl::host_accessor acc2 {buf_comp2};

            std::string id_err_list = "errors in id : ";

            bool eq = true;
            for(u32 i = 0; i < npart; i++){
                if(!shamsys::syclhelper::test_sycl_eq(acc1[i] , acc2[i])){
                    eq = false;
                    //id_err_list += std::to_string(i) + " ";
                }
            }

            if (eq) {
                shamtest::asserts().assert_bool("same content", eq);
            }else{
                shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
            }
        }
    }

}




TestStart(Unittest, "shamsys/comm/comm-buffer/syclbuffer-isend-irecv", isend_irecv_syclbuf, 2){

    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_comm_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_comm_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_comm_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_comm_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_comm_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_comm_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_comm_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_comm_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_comm_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_comm_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);
    
}



TestStart(Unittest, "shamsys/comm/comm-buffer/syclbuffer-isend-irecv_probe", isend_irecv_probe_syclbuf, 2){

    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_comm_probe_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_comm_probe_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_comm_probe_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);
    
}