#pragma once

#include "tests_handler.hpp"

inline void add_test_global(){

    add_test(
        "string_utils.hpp/format()", //name of the test
        false, // single process mode (no mpi)
        []{
            //printf("test 1 : rank=%d\n",world_rank);

            if(format("%s %d\n","ttt : ",10).compare("ttt :  10\n")){
                return Test_log("string_utils.hpp/format()",0,"");
            }else{
                return Test_log("string_utils.hpp/format()",1,"");
            }
        }
    );

}