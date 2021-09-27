#pragma once

#include "../sys/mpi_handler.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace unit_test{
    
    #define DECRIPTION_ASSERT_MAX_LEN 32
    struct Assert_report{
        char description[DECRIPTION_ASSERT_MAX_LEN];
        bool condition;
        unsigned int rank_emited;
        Assert_report(const char desc[DECRIPTION_ASSERT_MAX_LEN], bool cdt){

            std::memcpy(description,desc, sizeof(char)*DECRIPTION_ASSERT_MAX_LEN);
            rank_emited = world_rank;
            condition = cdt;
        }
    };


    struct Test_result{
        std::vector<Assert_report> assert_report_list;
        unsigned int succes_count;
        unsigned int assert_count;
        float succes_rate;
        std::string name;
        bool is_mpi;
    };



    //result output and log buffers
    inline bool is_current_test_mpi = false;
    inline std::vector<std::string> log_buffer;
    inline std::vector<Assert_report> assert_buffer;

    inline bool test_start(std::string test_name,bool is_mpi){

        

        is_current_test_mpi = is_mpi;

        if(! is_mpi){
            return world_rank == 0;
        }else{
            return true;
        }
    }


    
    inline void test_log(std::string str){



    }

    inline void test_assert(const char name[DECRIPTION_ASSERT_MAX_LEN], bool cdt){
        assert_buffer.push_back(Assert_report(name, cdt));
    }

    inline void test_end(){

        log_buffer.clear();

    }


    


}