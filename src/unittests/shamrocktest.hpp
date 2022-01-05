#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "../sys/mpi_handler.hpp"
#include "../aliases.hpp"
#include "../utils/time_utils.hpp"

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



#define Test_start(name, node_cnt) void test_func_##name (TestResults& t);\
void (*test_func_ptr_##name)(TestResults&) = test_func_##name;\
Test test_class_obj_##name (#name,node_cnt,test_func_ptr_##name);\
void test_func_##name (TestResults& __test_result_ref)



#define Test_assert(name,result) __test_result_ref.lst_assert.push_back({name, result, ""});
#define Test_assert_log(name,result,log) __test_result_ref.lst_assert.push_back({name, result, log});


//start test (name of the test, number of mpi node to use)
//Test_assert("assert name", succes boolean, "log corresponding to the assertion");



bool has_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }
    
    return false;
}

std::string_view get_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }
    
    return "";
}


inline int run_all_tests(int argc, char *argv[]){


    std::vector<std::string_view> args(argv + 1, argv + argc);



    const char* usage_str = R"%(Usage : ./shamrock_test2 [options]
Options :
  --help              Display this information
  --test-list         Display the list of tests available
  --run-only <name>   Run only the test with the specified name
    )%";


    if(has_option(args, "--help")){
        printf("%s",usage_str);return 0;
    }

    if(has_option(args, "--test-list")){
        for (unsigned int i = 0; i < test_name_lst.size(); i++) {
            if(test_node_count[i] == -1){
                printf("- [any] %-15s\n",test_name_lst[i].c_str());
            }else{
                printf("- [%03d] %-15s\n",test_node_count[i],test_name_lst[i].c_str());
            }
        }
        return 0;
    }

    bool run_only = false;
    std::string run_only_name = "";
    if(has_option(args, "--run-only")){
        run_only_name = get_option(args, "--run-only");
        run_only = true;
    }




    mpi_init();printf("\n");

    if(!run_only){
        printf("\n------------ Tests list --------------\n");
    }

    std::vector<u32> selected_tests = {};

    for (u32 i = 0; i < test_name_lst.size(); i++) {
        
        if(run_only){
            if(run_only_name.compare(test_name_lst[i]) ==0){
                selected_tests.push_back(i);
            }
        }else{
            bool any_node_cnt = (test_node_count[i] == -1);
            if(any_node_cnt || (test_node_count[i] == world_size)){
                selected_tests.push_back(i);
                if(any_node_cnt){
                    printf(" - [\033[;32many\033[0m] ");
                }else{
                    printf(" - [\033[;32m%03d\033[0m] ",test_node_count[i]);
                }
                
                std::cout << "\033[;32m"<<test_name_lst[i] <<  "\033[0m " <<std::endl;
            }else{
                printf(" - [\033[;31m%03d\033[0m] ",test_node_count[i]);
                std::cout << "\033[;31m"<<test_name_lst[i] <<  "\033[0m " <<std::endl;
            }
        }

    }

    if(!run_only){
        printf("--------------------------------------\n\n");
    }







    for (u32 i : selected_tests) {

        if(run_only){
            printf("running test : ",i+1);
        }else{
            printf("running test [%d/%d] : ",i+1,selected_tests.size());
        }

        bool any_node_cnt = test_node_count[i] == -1;
        if(any_node_cnt){
            printf("[any] ");
        }else{
            printf(" [%03d] ",test_node_count[i]);
        }

        std::cout << "\033[;34m"<<test_name_lst[i] <<  "\033[0m " <<std::endl;
        


        TestResults t(test_name_lst[i]);


        #ifdef MEM_TRACK_ENABLED
        u64 alloc_cnt_before_test = ptr_allocated.size();
        #endif
        mpi_barrier();
        Timer timer;
        timer.start();
        test_fct_lst[i](t);
        timer.end();
        mpi_barrier();

        
        #ifdef MEM_TRACK_ENABLED
        bool memory_clean = ptr_allocated.size() == alloc_cnt_before_test;

        if(memory_clean){
            std::cout << "        memory :  (\033[;32mClean\033[0m)\n";
        }else{
            std::cout << "        memory :  (\033[1;31m Dirty \033[0m)\n";
            print_state_alloc();
        }
        #endif

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

        std::cout << "       (" << timer.get_time_str() << ")" <<std::endl;

        std::cout << std::endl;

    }



    //recover result on node 0 and write to file if -o <outfile> specified




    mpi_close();


    return 0;
}