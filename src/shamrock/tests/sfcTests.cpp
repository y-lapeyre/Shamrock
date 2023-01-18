#include "shamrock/sfc/morton.hpp"
#include "shamtest/shamtest.hpp"

#include "aliases.hpp"

TestStart(Unittest, "shamrock/sfc/morton/min-max", mortonminmaxval, 1){


    using Morton32 = shamrock::sfc::MortonCodes<u32,3>;
    using Morton64 = shamrock::sfc::MortonCodes<u64,3>;

    u32 m_0_32 = Morton32::coord_to_morton(0, 0, 0);
    u32 m_max_32 = Morton32::coord_to_morton(1, 1, 1);

    u64 m_0_64 = Morton64::coord_to_morton(0, 0, 0);
    u64 m_max_64 = Morton64::coord_to_morton(1, 1, 1);


    shamtest::asserts().assert_bool("min morton 64 == b0", m_0_64 == 0x0);    
    shamtest::asserts().assert_bool("max morton 64 == b63x1", m_max_64 == 0x7fffffffffffffff);
    shamtest::asserts().assert_bool("min morton 32 == b0x0", m_0_32 == 0x0);    
    shamtest::asserts().assert_bool("max morton 32 == b30x1", m_max_32 == 0x3fffffff);
}



