// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/memory/memory.hpp"
#include "shamalgs/memory/serialize.hpp"
#include "shamalgs/random/random.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

template<class T>
inline void check_buf(std::string prefix, sycl::buffer<T> & b1, sycl::buffer<T> & b2){

    shamtest::asserts().assert_equal(prefix+std::string("same size"), b1.size(), b2.size());

    {
        sycl::host_accessor acc1 {b1};
        sycl::host_accessor acc2 {b2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for(u32 i = 0; i < b1.size(); i++){
            if(!shambase::vec_equals(acc1[i] , acc2[i])){
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

TestStart(Unittest, "shamalgs/memory/SerializeHelper", test_serialize_helper, 1){

    u32 n1 = 100;
    sycl::buffer<u8> buf_comp1 = shamalgs::random::mock_buffer<u8>(0x111, n1);

    f64_16 test_val = f64_16{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    u32 n2 = 100;
    sycl::buffer<u32_3> buf_comp2 = shamalgs::random::mock_buffer<u32_3>(0x121, n2);

    shamalgs::SerializeHelper ser;

    u64 bytelen = n1*sizeof(u8) + 16*sizeof(f64) + n2*3*sizeof(u32);

    ser.allocate(bytelen);
    ser.write_buf(buf_comp1, n1);
    ser.write(test_val);
    ser.write_buf(buf_comp2, n2);

    

    auto recov = ser.finalize();

    {
        sycl::buffer<u8> buf1 (n1);
        f64_16 val;
        sycl::buffer<u32_3> buf2 (n2);

        shamalgs::SerializeHelper ser2(std::move(recov));

        ser2.load_buf(buf1, n1);
        ser2.load(val);
        ser2.load_buf(buf2, n2);

        shamalgs::memory::print_buf(buf_comp1, n1, 16, "{} ");
        shamalgs::memory::print_buf(buf1, n1, 16, "{} ");

        shamtest::asserts().assert_bool("same", shambase::vec_equals(val , test_val));
        check_buf("buf 1", buf_comp1, buf1);
        check_buf("buf 2", buf_comp2, buf2);

    }
}