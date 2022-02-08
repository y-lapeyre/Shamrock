#pragma once

#include "../shamrocktest.hpp"

#include <random>

#include "../../scheduler/scheduler_mpi.hpp"

#include "../sys/test_sycl_interop_utils.hpp"


inline void check_patch_data_equal(TestResults &__test_result_ref, PatchData& p1, PatchData& p2){

    Test_assert("same_count_pos_s", p1.pos_s.size() == p2.pos_s.size());
    Test_assert("same_count_pos_d", p1.pos_d.size() == p2.pos_d.size());
    Test_assert("same_count_U1_s", p1.U1_s.size() == p2.U1_s.size());
    Test_assert("same_count_U1_d", p1.U1_d.size() == p2.U1_d.size());
    Test_assert("same_count_U3_s", p1.U3_s.size() == p2.U3_s.size());
    Test_assert("same_count_U3_d", p1.U3_d.size() == p2.U3_d.size());


    for (u32 i = 0; i < p1.pos_s.size(); i ++) {
        Test_assert("same value pos_s",test_eq3(p1.pos_s[i] , p2.pos_s[i] ));
    }
    
    for (u32 i = 0; i < p1.pos_d.size(); i ++) {
        Test_assert("same value pos_d",test_eq3(p1.pos_d[i] , p2.pos_d[i] ));
    }

    for (u32 i = 0; i < p1.U1_s.size(); i ++) {
        Test_assert("same value U1_s",p1.U1_s[i] == p2.U1_s[i] );
    }
    
    for (u32 i = 0; i < p1.U1_d.size(); i ++) {
        Test_assert("same value U1_d",p1.U1_d[i] == p2.U1_d[i] );
    }

    for (u32 i = 0; i < p1.U3_s.size(); i ++) {
        Test_assert("same value U3_s",test_eq3(p1.U3_s[i] , p2.U3_s[i] ));
    }
    
    for (u32 i = 0; i < p1.U3_d.size(); i ++) {
        Test_assert("same value U3_d",test_eq3(p1.U3_d[i] , p2.U3_d[i] ));
    }
}


inline PatchData gen_dummy_data(std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,1000);

    std::uniform_real_distribution<f64> distfd(-1e5,1e5);

    u32 num_part = distu64(eng);

    PatchData d;


    for (u32 i = 0 ; i < num_part; i++) {
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_s; ii ++) {
            d.pos_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
        }
        
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_d; ii ++) {
            d.pos_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_s; ii ++) {
            d.U1_s.push_back( f_s(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_d; ii ++) {
            d.U1_d.push_back( f_d(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_s; ii ++) {
            d.U3_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_d; ii ++) {
            d.U3_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
        }
    }

    return d;
}