#include "shamtest/shamtest.hpp"
#include <random>

#include "shamsys/Comm.hpp"
#include "shamsys/CommImplBuffer.hpp"
#include "shamsys/SyclHelper.hpp"


template<class T> void test_constructor_syclbuf(std::string prefix, std::mt19937 & eng, shamsys::comm::Protocol prot){

    std::uniform_real_distribution<f64> distval(-1.0F, 1.0F);

    u32 npart = 1e6;

    sycl::buffer<T> buf_comp (npart);

    {
        sycl::host_accessor acc {buf_comp};
        for(u32 i = 0; i < npart; i++){
            acc[i] = next_obj<T>(eng, distval);
        }
    }

    using namespace shamsys::comm;

    CommBuffer buf {buf_comp, prot};

    sycl::buffer<T> buf_comp2 = CommBuffer<sycl::buffer<T>>::convert(std::move(buf));


    shamtest::asserts().assert_equal(prefix+"same size", buf_comp.size(), buf_comp2.size());

    {
        sycl::host_accessor acc1 {buf_comp};
        sycl::host_accessor acc2 {buf_comp2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for(u32 i = 0; i < npart; i++){
            if(test_sycl_eq(acc1[i] , acc2[i])){
                eq = false;
                id_err_list += std::to_string(i) + " ";
            }
        }

        if (eq) {
            shamtest::asserts().assert_bool("same content", eq);
        }else{
            shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
        }
    }
}

TestStart(Unittest, "shamsys/comm/comm-buffer/syclbuffer", constructordestructor, 1){

    std::mt19937 eng(0x1111);

    using namespace shamsys::comm;
 
    test_constructor_syclbuf<f32   >("f32   : ",eng,CopyToHost);
    //test_constructor_syclbuf<f32_2 >("f32_2 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f32_3 >("f32_3 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f32_4 >("f32_4 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f32_8 >("f32_8 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f32_16>("f32_16: ",eng,CopyToHost);
    //test_constructor_syclbuf<f64   >("f64   : ",eng,CopyToHost);
    //test_constructor_syclbuf<f64_2 >("f64_2 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f64_3 >("f64_3 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f64_4 >("f64_4 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f64_8 >("f64_8 : ",eng,CopyToHost);
    //test_constructor_syclbuf<f64_16>("f64_16: ",eng,CopyToHost);
    //test_constructor_syclbuf<u32   >("u32   : ",eng,CopyToHost);
    //test_constructor_syclbuf<u64   >("u64   : ",eng,CopyToHost);
    
}