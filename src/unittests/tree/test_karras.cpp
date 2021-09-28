
#include "test_tree.hpp"

#include "../unit_test_handler.hpp"
#include "../../tree/karras_alg.hpp"
#include "../../sys/sycl_handler.hpp"
#include <vector>

void run_tests_karras_alg(){

    if(unit_test::test_start("tree/karras_alg.hpp", false)){


#if defined(PRECISION_MORTON_DOUBLE)
        
        

#else


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

            karras_alg(
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

        unit_test::test_assert("out_lchild_id[0] == 7", out_lchild_id[0] == 7);
        unit_test::test_assert("out_lchild_id[1] == 0", out_lchild_id[1] == 0);
        unit_test::test_assert("out_lchild_id[2] == 2", out_lchild_id[2] == 2);
        unit_test::test_assert("out_lchild_id[3] == 1", out_lchild_id[3] == 1);
        unit_test::test_assert("out_lchild_id[4] == 5", out_lchild_id[4] == 5);
        unit_test::test_assert("out_lchild_id[5] == 4", out_lchild_id[5] == 4);
        unit_test::test_assert("out_lchild_id[6] == 6", out_lchild_id[6] == 6);
        unit_test::test_assert("out_lchild_id[7] == 3", out_lchild_id[7] == 3);
        unit_test::test_assert("out_lchild_id[8] == 8", out_lchild_id[8] == 8);
        unit_test::test_assert("out_lchild_id[9] == 10", out_lchild_id[9] == 10);
        unit_test::test_assert("out_lchild_id[10] == 9", out_lchild_id[10] == 9);
        unit_test::test_assert("out_lchild_id[11] == 11", out_lchild_id[11] == 11);

#endif


    }unit_test::test_end();

}