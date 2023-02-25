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


template<class T> void test_constructor_syclbuf(std::string prefix, std::mt19937 & eng, shamsys::comm::Protocol prot){

    std::uniform_real_distribution<f64> distval(-1.0F, 1.0F);

    u32 npart = 1e5;

    sycl::buffer<T> buf_comp (npart);

    {
        sycl::host_accessor acc {buf_comp, sycl::write_only, sycl::no_init};
        for(u32 i = 0; i < npart; i++){
            acc[i] = shamsys::syclhelper::next_obj<T>(eng, distval);
        }
    }

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

    std::mt19937 eng(0x1111);

    using namespace shamsys::comm;
 
    test_constructor_syclbuf<f32   >("f32   : ",eng,CopyToHost);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",eng,CopyToHost);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",eng,CopyToHost);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",eng,CopyToHost);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",eng,CopyToHost);
    test_constructor_syclbuf<f32_16>("f32_16: ",eng,CopyToHost);
    test_constructor_syclbuf<f64   >("f64   : ",eng,CopyToHost);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",eng,CopyToHost);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",eng,CopyToHost);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",eng,CopyToHost);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",eng,CopyToHost);
    test_constructor_syclbuf<f64_16>("f64_16: ",eng,CopyToHost);
    test_constructor_syclbuf<u32   >("u32   : ",eng,CopyToHost);
    test_constructor_syclbuf<u64   >("u64   : ",eng,CopyToHost);


    test_constructor_syclbuf<f32   >("f32   : ",eng,DirectGPU);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",eng,DirectGPU);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",eng,DirectGPU);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",eng,DirectGPU);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",eng,DirectGPU);
    test_constructor_syclbuf<f32_16>("f32_16: ",eng,DirectGPU);
    test_constructor_syclbuf<f64   >("f64   : ",eng,DirectGPU);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",eng,DirectGPU);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",eng,DirectGPU);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",eng,DirectGPU);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",eng,DirectGPU);
    test_constructor_syclbuf<f64_16>("f64_16: ",eng,DirectGPU);
    test_constructor_syclbuf<u32   >("u32   : ",eng,DirectGPU);
    test_constructor_syclbuf<u64   >("u64   : ",eng,DirectGPU);

    test_constructor_syclbuf<f32   >("f32   : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f32_16>("f32_16: ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64   >("f64   : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<f64_16>("f64_16: ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<u32   >("u32   : ",eng,DirectGPUFlatten);
    test_constructor_syclbuf<u64   >("u64   : ",eng,DirectGPUFlatten);
    
}





















template<class T> void test_comm_syclbuf(std::string prefix, std::mt19937 & eng, shamsys::comm::Protocol prot){

    std::uniform_real_distribution<f64> distval(-1.0F, 1.0F);

    u32 npart = 1e4;

    sycl::buffer<T> buf_comp (npart);

    {
        sycl::host_accessor acc {buf_comp};
        for(u32 i = 0; i < npart; i++){
            acc[i] = shamsys::syclhelper::next_obj<T>(eng, distval);
        }
    }

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





TestStart(Unittest, "shamsys/comm/comm-buffer/syclbuffer-isend-irecv", isend_irecv_syclbuf, 2){

    std::mt19937 eng(0x1111);

    using namespace shamsys::comm;
 
    test_comm_syclbuf<f32   >("f32   : ",eng,CopyToHost);
    test_comm_syclbuf<f32_2 >("f32_2 : ",eng,CopyToHost);
    test_comm_syclbuf<f32_3 >("f32_3 : ",eng,CopyToHost);
    test_comm_syclbuf<f32_4 >("f32_4 : ",eng,CopyToHost);
    test_comm_syclbuf<f32_8 >("f32_8 : ",eng,CopyToHost);
    test_comm_syclbuf<f32_16>("f32_16: ",eng,CopyToHost);
    test_comm_syclbuf<f64   >("f64   : ",eng,CopyToHost);
    test_comm_syclbuf<f64_2 >("f64_2 : ",eng,CopyToHost);
    test_comm_syclbuf<f64_3 >("f64_3 : ",eng,CopyToHost);
    test_comm_syclbuf<f64_4 >("f64_4 : ",eng,CopyToHost);
    test_comm_syclbuf<f64_8 >("f64_8 : ",eng,CopyToHost);
    test_comm_syclbuf<f64_16>("f64_16: ",eng,CopyToHost);
    test_comm_syclbuf<u32   >("u32   : ",eng,CopyToHost);
    test_comm_syclbuf<u64   >("u64   : ",eng,CopyToHost);


    test_comm_syclbuf<f32   >("f32   : ",eng,DirectGPU);
    test_comm_syclbuf<f32_2 >("f32_2 : ",eng,DirectGPU);
    test_comm_syclbuf<f32_3 >("f32_3 : ",eng,DirectGPU);
    test_comm_syclbuf<f32_4 >("f32_4 : ",eng,DirectGPU);
    test_comm_syclbuf<f32_8 >("f32_8 : ",eng,DirectGPU);
    test_comm_syclbuf<f32_16>("f32_16: ",eng,DirectGPU);
    test_comm_syclbuf<f64   >("f64   : ",eng,DirectGPU);
    test_comm_syclbuf<f64_2 >("f64_2 : ",eng,DirectGPU);
    test_comm_syclbuf<f64_3 >("f64_3 : ",eng,DirectGPU);
    test_comm_syclbuf<f64_4 >("f64_4 : ",eng,DirectGPU);
    test_comm_syclbuf<f64_8 >("f64_8 : ",eng,DirectGPU);
    test_comm_syclbuf<f64_16>("f64_16: ",eng,DirectGPU);
    test_comm_syclbuf<u32   >("u32   : ",eng,DirectGPU);
    test_comm_syclbuf<u64   >("u64   : ",eng,DirectGPU);

    test_comm_syclbuf<f32   >("f32   : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f32_2 >("f32_2 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f32_3 >("f32_3 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f32_4 >("f32_4 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f32_8 >("f32_8 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f32_16>("f32_16: ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64   >("f64   : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64_2 >("f64_2 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64_3 >("f64_3 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64_4 >("f64_4 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64_8 >("f64_8 : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<f64_16>("f64_16: ",eng,DirectGPUFlatten);
    test_comm_syclbuf<u32   >("u32   : ",eng,DirectGPUFlatten);
    test_comm_syclbuf<u64   >("u64   : ",eng,DirectGPUFlatten);
    
}