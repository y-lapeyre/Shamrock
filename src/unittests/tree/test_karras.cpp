// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//#include "test_tree.hpp"


#include "core/tree/kernels/karras_alg.hpp"
#include "core/sys/sycl_handler.hpp"
#include <memory>
#include <vector>
#include "unittests/shamrocktest.hpp"


template<class mprec>
inline void test_karras_alg(TestResults & __test_result_ref){
    using u_morton = mprec;

    std::vector<u_morton> morton_list = {
        0x0,
        0x1,
        0x2,
        0x3,
        0x4,
        0x5,
        0x6,
        0x7,
        0x8,
        //0x9,
        //0xa,
        //0xb,
        0xc,
        0xd,
        0xe,
        0xf,
    };

    std::vector<u32> out_lchild_id(morton_list.size());
    std::vector<u32> out_rchild_id(morton_list.size());
    std::vector<u8 > out_lchild_flag(morton_list.size());
    std::vector<u8 > out_rchild_flag(morton_list.size());
    std::vector<u32> out_endrange(morton_list.size());

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton          = std::make_unique<sycl::buffer<u_morton>>(morton_list    .data(),morton_list    .size());
        std::unique_ptr<sycl::buffer<u32>     > out_buf_lchild_id   = std::make_unique<sycl::buffer<u32>     >(out_lchild_id  .data(),out_lchild_id  .size());
        std::unique_ptr<sycl::buffer<u32>     > out_buf_rchild_id   = std::make_unique<sycl::buffer<u32>     >(out_rchild_id  .data(),out_rchild_id  .size());
        std::unique_ptr<sycl::buffer<u8 >     > out_buf_lchild_flag = std::make_unique<sycl::buffer<u8 >     >(out_lchild_flag.data(),out_lchild_flag.size());
        std::unique_ptr<sycl::buffer<u8 >     > out_buf_rchild_flag = std::make_unique<sycl::buffer<u8 >     >(out_rchild_flag.data(),out_rchild_flag.size());
        std::unique_ptr<sycl::buffer<u32>     > out_buf_endrange    = std::make_unique<sycl::buffer<u32>     >(out_endrange   .data(),out_endrange   .size());

        sycl_karras_alg<u_morton>(
            sycl_handler::get_compute_queue(),
            morton_list.size()-1, 
            buf_morton, 
            out_buf_lchild_id,
            out_buf_rchild_id,
            out_buf_lchild_flag,
            out_buf_rchild_flag,
            out_buf_endrange
            );

    }

    Test_assert("out_lchild_id[0]  == 7",  out_lchild_id[0]  == 7);
    Test_assert("out_lchild_id[1]  == 0",  out_lchild_id[1]  == 0);
    Test_assert("out_lchild_id[2]  == 2",  out_lchild_id[2]  == 2);
    Test_assert("out_lchild_id[3]  == 1",  out_lchild_id[3]  == 1);
    Test_assert("out_lchild_id[4]  == 5",  out_lchild_id[4]  == 5);
    Test_assert("out_lchild_id[5]  == 4",  out_lchild_id[5]  == 4);
    Test_assert("out_lchild_id[6]  == 6",  out_lchild_id[6]  == 6);
    Test_assert("out_lchild_id[7]  == 3",  out_lchild_id[7]  == 3);
    Test_assert("out_lchild_id[8]  == 8",  out_lchild_id[8]  == 8);
    Test_assert("out_lchild_id[9]  == 10", out_lchild_id[9]  == 10);
    Test_assert("out_lchild_id[10] == 9",  out_lchild_id[10] == 9);
    Test_assert("out_lchild_id[11] == 11", out_lchild_id[11] == 11);

    Test_assert("out_rchild_id[0]  == 8",  out_rchild_id[0]  == 8);
    Test_assert("out_rchild_id[1]  == 1",  out_rchild_id[1]  == 1);
    Test_assert("out_rchild_id[2]  == 3",  out_rchild_id[2]  == 3);
    Test_assert("out_rchild_id[3]  == 2",  out_rchild_id[3]  == 2);
    Test_assert("out_rchild_id[4]  == 6",  out_rchild_id[4]  == 6);
    Test_assert("out_rchild_id[5]  == 5",  out_rchild_id[5]  == 5);
    Test_assert("out_rchild_id[6]  == 7",  out_rchild_id[6]  == 7);
    Test_assert("out_rchild_id[7]  == 4",  out_rchild_id[7]  == 4);
    Test_assert("out_rchild_id[8]  == 9",  out_rchild_id[8]  == 9);
    Test_assert("out_rchild_id[9]  == 11", out_rchild_id[9]  == 11);
    Test_assert("out_rchild_id[10] == 10", out_rchild_id[10] == 10);
    Test_assert("out_rchild_id[11] == 12", out_rchild_id[11] == 12);


    Test_assert("out_lchild_flag[0]  == 0",  out_lchild_flag[0]  == 0);
    Test_assert("out_lchild_flag[1]  == 1",  out_lchild_flag[1]  == 1);
    Test_assert("out_lchild_flag[2]  == 1",  out_lchild_flag[2]  == 1);
    Test_assert("out_lchild_flag[3]  == 0",  out_lchild_flag[3]  == 0);
    Test_assert("out_lchild_flag[4]  == 0",  out_lchild_flag[4]  == 0);
    Test_assert("out_lchild_flag[5]  == 1",  out_lchild_flag[5]  == 1);
    Test_assert("out_lchild_flag[6]  == 1",  out_lchild_flag[6]  == 1);
    Test_assert("out_lchild_flag[7]  == 0",  out_lchild_flag[7]  == 0);
    Test_assert("out_lchild_flag[8]  == 1",  out_lchild_flag[8]  == 1);
    Test_assert("out_lchild_flag[9]  == 0",  out_lchild_flag[9]  == 0);
    Test_assert("out_lchild_flag[10] == 1",  out_lchild_flag[10] == 1);
    Test_assert("out_lchild_flag[11] == 1",  out_lchild_flag[11] == 1);

    Test_assert("out_rchild_flag[0]  == 0",  out_rchild_flag[0]  == 0);
    Test_assert("out_rchild_flag[1]  == 1",  out_rchild_flag[1]  == 1);
    Test_assert("out_rchild_flag[2]  == 1",  out_rchild_flag[2]  == 1);
    Test_assert("out_rchild_flag[3]  == 0",  out_rchild_flag[3]  == 0);
    Test_assert("out_rchild_flag[4]  == 0",  out_rchild_flag[4]  == 0);
    Test_assert("out_rchild_flag[5]  == 1",  out_rchild_flag[5]  == 1);
    Test_assert("out_rchild_flag[6]  == 1",  out_rchild_flag[6]  == 1);
    Test_assert("out_rchild_flag[7]  == 0",  out_rchild_flag[7]  == 0);
    Test_assert("out_rchild_flag[8]  == 0",  out_rchild_flag[8]  == 0);
    Test_assert("out_rchild_flag[9]  == 0",  out_rchild_flag[9]  == 0);
    Test_assert("out_rchild_flag[10] == 1",  out_rchild_flag[10] == 1);
    Test_assert("out_rchild_flag[11] == 1",  out_rchild_flag[11] == 1);

    Test_assert("out_endrange[0]  == 12", out_endrange[0]  == 12);
    Test_assert("out_endrange[1]  == 0",  out_endrange[1]  == 0);
    Test_assert("out_endrange[2]  == 3",  out_endrange[2]  == 3);
    Test_assert("out_endrange[3]  == 0",  out_endrange[3]  == 0);
    Test_assert("out_endrange[4]  == 7",  out_endrange[4]  == 7);
    Test_assert("out_endrange[5]  == 4",  out_endrange[5]  == 4);
    Test_assert("out_endrange[6]  == 7",  out_endrange[6]  == 7);
    Test_assert("out_endrange[7]  == 0",  out_endrange[7]  == 0);
    Test_assert("out_endrange[8]  == 12", out_endrange[8]  == 12);
    Test_assert("out_endrange[9]  == 12", out_endrange[9]  == 12);
    Test_assert("out_endrange[10] == 9",  out_endrange[10] == 9);
    Test_assert("out_endrange[11] == 12", out_endrange[11] == 12);
}


Test_start("tree::kernels::",karras_alg32,1){
    test_karras_alg<u32>(__test_result_ref);
}

Test_start("tree::kernels::",karras_alg64,1){
    test_karras_alg<u64>(__test_result_ref);
}