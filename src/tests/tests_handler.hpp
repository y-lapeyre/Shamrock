#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "../sys/mpi_handler.hpp"


class Test_log{ 
public:
    std::string test_name;

    float succes_rate;
    std::string log;

    Test_log(std::string test_name, float succes_rate,std::string log){
        this->test_name = test_name;
        this->succes_rate = succes_rate;
        this->log = log;
    }

};

inline std::vector<Test_log> test_log_list;



/**
 * @brief class to wrapp test
 * 
 */
class Test{
public:
    std::string name;
    bool mpi_test = false;
    Test_log (*test_func)();

    Test(std::string name, bool mpi_test ,Test_log (*test_func)()){
        this->name = name;
        this->test_func = test_func;
        this->mpi_test = mpi_test;
    }

    /**
     * @brief note : not designed to be thread safe
     */
    void run_test(){

        if(mpi_test){
            Test_log ret = test_func();
            if(world_rank == 0){
                test_log_list.push_back(ret);
            }
        }else{
            if(world_rank == 0){
                Test_log ret = test_func();
                test_log_list.push_back(ret);
            }
        }
        
    }

    
};

inline std::vector<Test> test_list;

/**
 * @brief add a test to the test list
 * 
 * @param name test name
 * @param mpi_test does the test should run with multiple MPI processes
 * @param test_func lambda function that will perfom the test and return the Test log
 */
inline void add_test(std::string name, bool mpi_test ,Test_log (*test_func)() ){
    test_list.push_back(Test(name, mpi_test ,test_func));
}




inline void run_tests(){
    mpi_barrier();

    unsigned int test_count = test_list.size();

    for(unsigned int i = 0; i < test_count; i++){
        if(world_rank == 0){
            //printf("\r");
            printf("[%d/%d] %s",i+1,test_count,test_list[i].name.c_str());
            fflush(stdout);

            if(i+1 == test_count){
                printf("\n");
            }
        }
        
        test_list[i].run_test();
        mpi_barrier();
    }

    

}

inline void show_logs(){
    if(world_rank == 0){

        const char* format_char_desc = "| %13s | %13s% | %13s |\n";
        const char* format_char      = "| %13s | %13d% | %13s |\n";

        printf(format_char_desc,"test name","succes rate","log");

        for(Test_log t : test_log_list){
            printf(format_char,t.test_name.c_str(),(int)t.succes_rate*100,t.log.c_str());
        }
    }
}