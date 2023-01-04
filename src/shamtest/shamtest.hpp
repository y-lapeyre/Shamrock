#include "aliases.hpp"

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


