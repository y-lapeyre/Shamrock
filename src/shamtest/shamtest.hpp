// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "details/Test.hpp"

/**
 * @brief namespace containing stuff related to the test library
 * 
 */
namespace shamtest {

    /**
     * @brief implementation details of the test library
     * 
     */
    namespace details {

        /**
         * @brief Static init vector containing the list of all the tests in the code
         * see : programming guide : Static init function registering
         */
        inline std::vector<Test> static_init_vec_tests{};

        /**
         * @brief helper class to statically register tests
         * 
         */
        struct TestStaticInit {
            inline explicit TestStaticInit(Test t) { static_init_vec_tests.push_back(std::move(t)); }
        };

        /**
         * @brief the test currently running
         * 
         */
        extern TestResult current_test;

    } // namespace details

    /**
     * @brief run all the tests
     * 
     * @param argc main arg
     * @param argv  main arg
     * @param run_bench run benchmarks ?
     * @param run_analysis run analysis tests ?
     * @param run_unittest run unittests ?
     * @return int 
     */
    int run_all_tests(int argc, char *argv[], bool run_bench, bool run_analysis, bool run_unittest);

    /**
     * @brief current test asserts
     * 
     * @return shamtest::details::TestAssertList& reference to the test asserts
     */
    inline shamtest::details::TestAssertList &asserts() {
        return shamtest::details::current_test.asserts;
    };

    /**
     * @brief current test data
     * 
     * @return shamtest::details::TestAssertList& reference to the test datas
     */
    inline shamtest::details::TestDataList &test_data() {
        return shamtest::details::current_test.test_data;
    };
} // namespace shamtest

/**
 * @brief Macro to declare a test
 * 
 */
#define TestStart(type, name, func_name, node_cnt)                                                 \
    void test_func_##func_name();                                                                  \
    void (*test_func_ptr_##func_name)() = test_func_##func_name;                                   \
    shamtest::details::TestStaticInit test_class_obj_##func_name(shamtest::details::Test{          \
        type, name, node_cnt, test_func_ptr_##func_name});                                         \
    void test_func_##func_name()

#define _Assert(a,b,prec) \
    shamtest::asserts().assert_bool(#a, a);

#define _AssertEqual(a,b) \
    shamtest::asserts().assert_equal(#a "==" #b, a, b);

#define _AssertFloatEqual(a,b,prec) \
    shamtest::asserts().assert_float_equal(#a  " ==("#prec") " #b, a, b, prec);