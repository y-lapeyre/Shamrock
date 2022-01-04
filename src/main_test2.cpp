#include "mem_track.hpp"

#include "aliases.hpp"
#include <cstdio>
#include <map>
#include <ostream>
#include <vector>
#include <string>
#include <iostream>
#include <functional>

#include <CL/sycl.hpp>










std::vector<int> test_node_count;
std::vector<std::string> test_name_lst;

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
std::vector<void (*)(TestResults &)> test_fct_lst;


class Test{public:
    

    Test(std::string test_name,int node_used,void (*test_func)(TestResults &) ){
        test_name_lst.push_back(test_name);
        test_node_count.push_back(node_used);
        test_fct_lst.push_back(test_func);
    }

}; 



#define Test_start(name, node_cnt) void test_func_##name (TestResults& t);\
void (*test_func_ptr_##name)(TestResults&) = test_func_##name;\
Test test_class_obj_##name (#name,0,test_func_ptr_##name);\
void test_func_##name (TestResults& __test_result_ref)







int run_all_tests(){


    for (unsigned int i = 0; i < test_name_lst.size(); i++) {
        printf("running test [%d/%d] : ",i+1,test_name_lst.size());
        std::cout << "\033[;34m"<<test_name_lst[i] <<  "\033[0m"<<std::endl;
        


        TestResults t(test_name_lst[i]);


        test_fct_lst[i](t);

        
        
        bool memory_clean = ptr_allocated.size() == 0;

        if(memory_clean){
            std::cout << "        memory :  (\033[;32mClean\033[0m)\n";
        }else{
            std::cout << "        memory :  (\033[1;31m Dirty \033[0m)\n";
            print_state_alloc();
        }

        //printf("    assertion list :\n");
        for(unsigned int j = 0; j < t.lst_assert.size(); j++){
            printf("        [%d/%d] : ",j+1,t.lst_assert.size());
            printf("%-20s",t.lst_assert[j].assert_name.c_str());
            
            if(t.lst_assert[j].success){
                std::cout << "  (\033[;32mSucces\033[0m)\n";
            }else{
                std::cout << "  (\033[1;31m Fail \033[0m)\n";
            }
            
            //std::cout << "            logs : " << t.lst_assert[j].log << "\n";
        }

        std::cout << std::endl;

    }


    return 0;
}





/*
Test_start(test_sycl,0){
    auto def_sel = sycl::default_selector();

    sycl::queue queue = sycl::queue(def_sel);

    float* rho = new float[10];
    sycl::buffer<float>* buf_rho = new sycl::buffer<float>(rho,10);

    queue.submit( [&](sycl::handler & cgh){
        auto rho = buf_rho->get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class Compute_Flux>(
            sycl::range<1>(10), 
            [=](sycl::item<1> item){
                unsigned int i = item.get_linear_id();

                    rho[i] = i;

                }
        );

    });

    delete buf_rho;

    std::cout << rho[0] << " " << rho[1] << " " << rho[2] << " " << rho[3] << " " << rho[4] << std::endl;

    delete [] rho;
}
*/



#define Test_assert(name,result) __test_result_ref.lst_assert.push_back({name, result, ""});
#define Test_assert_log(name,result,log) __test_result_ref.lst_assert.push_back({name, result, log});


//start test (name of the test, number of mpi node to use)
//Test_assert("assert name", succes boolean, "log corresponding to the assertion");



Test_start(intmult,0){
    int a = 3;
    a*=2;
    Test_assert("int multiplication", a==6);
}

Test_start(intdiv,0){
    int a = 6;
    a/=2;

    Test_assert("int division", a==3);
}

Test_start(multiple_asserts,0){

    int t[]{0,1,2,3};
    for (int i = 0; i<4; i++) {
        Test_assert("for loop assert", t[i]==i);
    }

}



Test_start(test_overload_new,0){

    int* a =  (int*) log_new(new int(0), log_alloc_ln);


    *a = 1;

    std::cout << *a << std::endl;



    delete a;
    //log_delete(a);

}

int main(void){

    printf("%s\n",git_info_str.c_str());

    return run_all_tests();
}
