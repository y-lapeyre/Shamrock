#pragma once

#include "../sys/mpi_handler.hpp"
#include "../aliases.hpp"
#include "unit_test_handler.hpp"
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

    struct Test_report{
        std::vector<Assert_report> assert_report_list;
        std::vector<std::string> logs;
        unsigned int succes_count;
        unsigned int assert_count;
        float succes_rate;
        std::string name;
        bool is_mpi;
    };




    inline std::vector<Test_report> tests_results;


    bool test_start(std::string test_name,bool is_mpi);
    
    void test_log(std::string str);

    bool test_assert(const char name[DECRIPTION_ASSERT_MAX_LEN], bool cdt);

    inline bool test_assert(const char name[DECRIPTION_ASSERT_MAX_LEN], u32 int1, u32 int2){
        return test_assert(name, int1 == int2);
    };

    void test_end();

    void print_test_results();

    void write_test_results(std::string filename);
    


}