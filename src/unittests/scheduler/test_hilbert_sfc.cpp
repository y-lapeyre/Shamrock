#include "unittests/shamrocktest.hpp"

#include "aliases.hpp"

#include "sfc/hilbert.hpp"

Test_start("sfc::", hilbert, 1) {
    
    Test_assert("compute_hilbert_index(5, 10, 20, 5) == 7865", compute_hilbert_index_3d<5>(5, 10, 20) == 7865);
    Test_assert("compute_hilbert_index(0,0,0, 21) == 7865", compute_hilbert_index_3d<21>(0,0,0) == 0);
    Test_assert("compute_hilbert_index(hilbert_box21_sz,0,0, 21)", compute_hilbert_index_3d<21>(hilbert_box21_sz,0,0) == 9223372036854775807);

}