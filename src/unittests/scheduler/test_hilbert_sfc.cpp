#include "../shamrocktest.hpp"

#include "../../aliases.hpp"

#include "../../scheduler/hilbertsfc.hpp"

Test_start("sfc::", hilbert, 1) {
    
    Test_assert("compute_hilbert_index(5, 10, 20, 5) == 7865", compute_hilbert_index<5>(5, 10, 20) == 7865);
    Test_assert("compute_hilbert_index(0,0,0, 21) == 7865", compute_hilbert_index<21>(0,0,0) == 0);
    Test_assert("compute_hilbert_index(2097152-1,0,0, 21)", compute_hilbert_index<21>(2097152-1,0,0) == 9223372036854775807);

}