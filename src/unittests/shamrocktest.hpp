// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

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


//%Impl status : Good


#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
//#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_handler.hpp"
#include "aliases.hpp"
#include "core/utils/time_utils.hpp"

[[deprecated]]
inline std::vector<int> test_node_count;
[[deprecated]]
inline std::vector<std::string> test_name_lst;


struct [[deprecated]] TestAssert {
    std::string assert_name;
    bool success;
    std::string log;
};


class [[deprecated]] TestResults{public:
    std::string test_name;
    std::vector<TestAssert> lst_assert = std::vector<TestAssert>(0);

    TestResults(std::string partest_name){
        test_name = partest_name;
    }
};

[[deprecated]]
inline std::vector<void (*)(TestResults &)> test_fct_lst;


class [[deprecated]] Test{public:
    

    Test(std::string test_name,int node_used,void (*test_func)(TestResults &) ){
        test_name_lst.push_back(test_name);
        test_node_count.push_back(node_used);
        test_fct_lst.push_back(test_func);
    }

}; 



#define Test_start(group,name, node_cnt) void test_func_##name (TestResults& t);\
void (*test_func_ptr_##name)(TestResults&) = test_func_##name;\
Test test_class_obj_##name (group ":" #name,node_cnt,test_func_ptr_##name);\
void test_func_##name (TestResults& __test_result_ref)



#define Test_assert(name,result)  \
{\
    std::string log = "";\
    if(!(result)){\
        log = __LOC_PREFIX__ + std::string( "\n    -> \"" #result "\" should be true");\
    }\
    __test_result_ref.lst_assert.push_back({name, result, log});\
}

#define Test_assert_log(name,result,log) __test_result_ref.lst_assert.push_back({name, result, log});


//start test (name of the test, number of mpi node to use)
//Test_assert("assert name", succes boolean, "log corresponding to the assertion");

[[deprecated]]
void run_py_script(std::string pysrc);

int run_all_tests(int argc, char *argv[]);




enum TestType{
    Benchmark,Analysis,Unittest
};

namespace impl::shamrocktest {

    

    struct TestAssert{
        bool value;
        std::string name;
        std::string comment;

        std::string serialize();
        
    };

    struct DataNode{
        std::string name;
        std::vector<f64> data;


        std::string serialize();
    };

    struct TestData{
        std::string dataset_name;
        std::vector<DataNode> dataset;

        inline void add_data(std::string name, const std::vector<f64> & v){
            std::vector<f64> new_vec;
            for(f64 f : v){
                new_vec.push_back(f);
            }
            dataset.push_back(DataNode{std::move(name),std::move(new_vec)});
        }


        std::string serialize();
    }; 

    struct TestAssertList{
        std::vector<TestAssert> asserts;

        //define member function here
        //to register asserts


        inline void assert_bool(std::string assert_name,bool v){
            asserts.push_back(TestAssert{v,std::move(assert_name),""});
        }

        template<class T>
        inline void assert_equal(std::string assert_name,T a, T b){

            bool t = a==b;
            std::string comment = "";

            if(!t){
                comment = "left="+std::to_string(a) + " right=" + std::to_string(b);
            }

            asserts.push_back(TestAssert{t,std::move(assert_name),comment});
        }


        
        inline void assert_float_equal(std::string assert_name,f64 a, f64 b, f64 eps){
            f64 diff = sycl::fabs(a - b);

            bool t = diff < eps;
            std::string comment = "";

            if(!t){
                comment = "left="+std::to_string(a) + " right=" + std::to_string(b) + " diff="+ std::to_string(diff);
            }

            asserts.push_back(TestAssert{t,std::move(assert_name),comment});
        }

        inline void assert_add_comment(std::string assert_name,bool v,std::string comment){
            asserts.push_back(TestAssert{v,std::move(assert_name),std::move(comment)});
        }

        std::string serialize();
    };

    struct TestDataList{
        std::vector<TestData> test_data;

        //define member function here
        //to register test data

        [[nodiscard]]
        inline TestData & new_dataset(std::string name){
            test_data.push_back(TestData{std::move(name),{}});
            return test_data.back();
        }


        std::string serialize();
    };

    struct TestResult{
        TestType type;
        std::string name;
        u32 world_rank;
        TestAssertList asserts;
        TestDataList test_data;

        inline TestResult (const TestType & type, std::string  name, const u32 & world_rank) :
        type(type), name(std::move(name)), world_rank(world_rank),asserts(),test_data()
        {}


        std::string serialize();
        
    };

    struct Test{
        TestType type;
        std::string name;
        i32 node_count;
        void (*test_functor)();


        inline Test (const TestType & type, std::string  name, const i32 & node_count,void (*func)() ) :
        type(type), name(std::move(name)),node_count(node_count), test_functor(func){}

        TestResult run();
    };

    inline std::vector<Test> static_init_vec_tests{};

    struct TestStaticInit{
        inline explicit TestStaticInit(Test t){
            static_init_vec_tests.push_back(std::move(t));
        }
    };

    extern TestResult current_test;

}

namespace shamrock::test {
    int run_all_tests(int argc, char *argv[], bool run_bench,bool run_analysis, bool run_unittest);

    inline impl::shamrocktest::TestAssertList & asserts(){return impl::shamrocktest::current_test.asserts;};
    inline impl::shamrocktest::TestDataList & test_data(){return impl::shamrocktest::current_test.test_data;};
}


#define TestStart(type,name,func_name, node_cnt) void test_func_##func_name ();\
void (*test_func_ptr_##func_name)() = test_func_##func_name;\
impl::shamrocktest::TestStaticInit test_class_obj_##func_name (impl::shamrocktest::Test{type,name,node_cnt,test_func_ptr_##func_name});\
void test_func_##func_name ()


