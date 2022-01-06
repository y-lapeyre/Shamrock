#include "../shamrocktest.hpp"

#include "../../tree/local/morton.hpp"

Test_start("morton::",min_max_value,1){

    u_morton m_0 = morton::xyz_to_morton(0, 0, 0);
    u_morton m_max = morton::xyz_to_morton(1, 1, 1);

    #if defined(PRECISION_MORTON_DOUBLE)
    Test_assert("min morton == b0", m_0 == 0x0);    
    Test_assert("max morton == b30x1", m_max == 0x7fffffffffffffff);
    #else
    Test_assert("min morton == b0x0", m_0 == 0x0);    
    Test_assert("max morton == b63x1", m_max == 0x3fffffff);
    #endif

}