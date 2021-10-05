#include "test_tree.hpp"


#include "../unit_test_handler.hpp"

#include "../../tree/local/morton.hpp"

void run_tests_morton(){

    if(unit_test::test_start("tree/morton.hpp", false)){

        u_morton m_0 = morton::xyz_to_morton(0, 0, 0);
        u_morton m_max = morton::xyz_to_morton(1, 1, 1);



        #if defined(PRECISION_MORTON_DOUBLE)
        unit_test::test_assert("min morton == b0", m_0 == 0x0);    
        unit_test::test_assert("max morton == b30x1", m_max == 0x7fffffffffffffff);
        #else
        unit_test::test_assert("min morton == b0x0", m_0 == 0x0);    
        unit_test::test_assert("max morton == b63x1", m_max == 0x3fffffff);
        #endif


    }unit_test::test_end();

}