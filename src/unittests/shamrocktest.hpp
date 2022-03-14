/**
 * @file shamrocktest.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "sys/mpi_handler.hpp"
#include "sys/sycl_handler.hpp"
#include "aliases.hpp"
#include "utils/time_utils.hpp"

inline std::vector<int> test_node_count;
inline std::vector<std::string> test_name_lst;

struct TestAssert{
    std::string assert_name;
    bool success;
    std::string log;
};

class TestResults{public:
    std::string test_name;
    std::vector<TestAssert> lst_assert = std::vector<TestAssert>(0);

    TestResults(std::string partest_name){
        test_name = partest_name;
    }
};

inline std::vector<void (*)(TestResults &)> test_fct_lst;


class Test{public:
    

    Test(std::string test_name,int node_used,void (*test_func)(TestResults &) ){
        test_name_lst.push_back(test_name);
        test_node_count.push_back(node_used);
        test_fct_lst.push_back(test_func);
    }

}; 



#define Test_start(group,name, node_cnt) void test_func_##name (TestResults& t);\
void (*test_func_ptr_##name)(TestResults&) = test_func_##name;\
Test test_class_obj_##name (group #name,node_cnt,test_func_ptr_##name);\
void test_func_##name (TestResults& __test_result_ref)



#define Test_assert(name,result) __test_result_ref.lst_assert.push_back({name, result, ""});
#define Test_assert_log(name,result,log) __test_result_ref.lst_assert.push_back({name, result, log});


//start test (name of the test, number of mpi node to use)
//Test_assert("assert name", succes boolean, "log corresponding to the assertion");



int run_all_tests(int argc, char *argv[]);