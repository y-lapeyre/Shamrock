
//#include "test_tree.hpp"


#include "tree/local/kernels/karras_alg.hpp"
#include "sys/sycl_handler.hpp"
#include <vector>
/*
void run_tests_karras_alg(){

    //for(u32 i = 0 ; i < )

    if(unit_test::test_start("tree/karras_alg.hpp", false)){

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
            sycl::buffer<u_morton> buf_morton(morton_list);

            sycl::buffer<u32> out_buf_lchild_id(out_lchild_id);
            sycl::buffer<u32> out_buf_rchild_id(out_rchild_id);
            sycl::buffer<u8 > out_buf_lchild_flag(out_lchild_flag);
            sycl::buffer<u8 > out_buf_rchild_flag(out_rchild_flag);
            sycl::buffer<u32> out_buf_endrange(out_endrange);

            sycl_karras_alg(
                queue,
                morton_list.size()-1, 
                &buf_morton, 
                &out_buf_lchild_id,
                &out_buf_rchild_id,
                &out_buf_lchild_flag,
                &out_buf_rchild_flag,
                &out_buf_endrange
                );

        }

        unit_test::test_assert("out_lchild_id[0]  == 7",  out_lchild_id[0]  , 7);
        unit_test::test_assert("out_lchild_id[1]  == 0",  out_lchild_id[1]  , 0);
        unit_test::test_assert("out_lchild_id[2]  == 2",  out_lchild_id[2]  , 2);
        unit_test::test_assert("out_lchild_id[3]  == 1",  out_lchild_id[3]  , 1);
        unit_test::test_assert("out_lchild_id[4]  == 5",  out_lchild_id[4]  , 5);
        unit_test::test_assert("out_lchild_id[5]  == 4",  out_lchild_id[5]  , 4);
        unit_test::test_assert("out_lchild_id[6]  == 6",  out_lchild_id[6]  , 6);
        unit_test::test_assert("out_lchild_id[7]  == 3",  out_lchild_id[7]  , 3);
        unit_test::test_assert("out_lchild_id[8]  == 8",  out_lchild_id[8]  , 8);
        unit_test::test_assert("out_lchild_id[9]  == 10", out_lchild_id[9]  , 10);
        unit_test::test_assert("out_lchild_id[10] == 9",  out_lchild_id[10] , 9);
        unit_test::test_assert("out_lchild_id[11] == 11", out_lchild_id[11] , 11);

        unit_test::test_assert("out_rchild_id[0]  == 8",  out_rchild_id[0]  , 8);
        unit_test::test_assert("out_rchild_id[1]  == 1",  out_rchild_id[1]  , 1);
        unit_test::test_assert("out_rchild_id[2]  == 3",  out_rchild_id[2]  , 3);
        unit_test::test_assert("out_rchild_id[3]  == 2",  out_rchild_id[3]  , 2);
        unit_test::test_assert("out_rchild_id[4]  == 6",  out_rchild_id[4]  , 6);
        unit_test::test_assert("out_rchild_id[5]  == 5",  out_rchild_id[5]  , 5);
        unit_test::test_assert("out_rchild_id[6]  == 7",  out_rchild_id[6]  , 7);
        unit_test::test_assert("out_rchild_id[7]  == 4",  out_rchild_id[7]  , 4);
        unit_test::test_assert("out_rchild_id[8]  == 9",  out_rchild_id[8]  , 9);
        unit_test::test_assert("out_rchild_id[9]  == 11", out_rchild_id[9]  , 11);
        unit_test::test_assert("out_rchild_id[10] == 10", out_rchild_id[10] , 10);
        unit_test::test_assert("out_rchild_id[11] == 12", out_rchild_id[11] , 12);


        unit_test::test_assert("out_lchild_flag[0]  == 0",  out_lchild_flag[0]  , 0);
        unit_test::test_assert("out_lchild_flag[1]  == 1",  out_lchild_flag[1]  , 1);
        unit_test::test_assert("out_lchild_flag[2]  == 1",  out_lchild_flag[2]  , 1);
        unit_test::test_assert("out_lchild_flag[3]  == 0",  out_lchild_flag[3]  , 0);
        unit_test::test_assert("out_lchild_flag[4]  == 0",  out_lchild_flag[4]  , 0);
        unit_test::test_assert("out_lchild_flag[5]  == 1",  out_lchild_flag[5]  , 1);
        unit_test::test_assert("out_lchild_flag[6]  == 1",  out_lchild_flag[6]  , 1);
        unit_test::test_assert("out_lchild_flag[7]  == 0",  out_lchild_flag[7]  , 0);
        unit_test::test_assert("out_lchild_flag[8]  == 1",  out_lchild_flag[8]  , 1);
        unit_test::test_assert("out_lchild_flag[9]  == 0",  out_lchild_flag[9]  , 0);
        unit_test::test_assert("out_lchild_flag[10] == 1",  out_lchild_flag[10] , 1);
        unit_test::test_assert("out_lchild_flag[11] == 1",  out_lchild_flag[11] , 1);

        unit_test::test_assert("out_rchild_flag[0]  == 0",  out_rchild_flag[0]  , 0);
        unit_test::test_assert("out_rchild_flag[1]  == 1",  out_rchild_flag[1]  , 1);
        unit_test::test_assert("out_rchild_flag[2]  == 1",  out_rchild_flag[2]  , 1);
        unit_test::test_assert("out_rchild_flag[3]  == 0",  out_rchild_flag[3]  , 0);
        unit_test::test_assert("out_rchild_flag[4]  == 0",  out_rchild_flag[4]  , 0);
        unit_test::test_assert("out_rchild_flag[5]  == 1",  out_rchild_flag[5]  , 1);
        unit_test::test_assert("out_rchild_flag[6]  == 1",  out_rchild_flag[6]  , 1);
        unit_test::test_assert("out_rchild_flag[7]  == 0",  out_rchild_flag[7]  , 0);
        unit_test::test_assert("out_rchild_flag[8]  == 0",  out_rchild_flag[8]  , 0);
        unit_test::test_assert("out_rchild_flag[9]  == 0",  out_rchild_flag[9]  , 0);
        unit_test::test_assert("out_rchild_flag[10] == 1",  out_rchild_flag[10] , 1);
        unit_test::test_assert("out_rchild_flag[11] == 1",  out_rchild_flag[11] , 1);


        unit_test::test_assert("out_endrange[0]  == 12", out_endrange[0]  == 12);
        unit_test::test_assert("out_endrange[1]  == 0",  out_endrange[1]  == 0);
        unit_test::test_assert("out_endrange[2]  == 3",  out_endrange[2]  == 3);
        unit_test::test_assert("out_endrange[3]  == 0",  out_endrange[3]  == 0);
        unit_test::test_assert("out_endrange[4]  == 7",  out_endrange[4]  == 7);
        unit_test::test_assert("out_endrange[5]  == 4",  out_endrange[5]  == 4);
        unit_test::test_assert("out_endrange[6]  == 7",  out_endrange[6]  == 7);
        unit_test::test_assert("out_endrange[7]  == 0",  out_endrange[7]  == 0);
        unit_test::test_assert("out_endrange[8]  == 12", out_endrange[8]  == 12);
        unit_test::test_assert("out_endrange[9]  == 12", out_endrange[9]  == 12);
        unit_test::test_assert("out_endrange[10] == 9",  out_endrange[10] == 9);
        unit_test::test_assert("out_endrange[11] == 12", out_endrange[11] == 12);


    }unit_test::test_end();

}
*/