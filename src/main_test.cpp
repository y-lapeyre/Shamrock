/**
 * @file main_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-08-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

//#include "tests/tests_handler.hpp"
#include "tests/unit_test_handler.hpp"

// #include "tests/test_global.hpp"
// #include "tests/test_sph.hpp"
// #include "tests/test_amr.hpp"

#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <cstdlib>

#include "sys/mpi_handler.hpp"

#include "utils/string_utils.hpp"

#include "tree/morton.hpp"
#include "tree/radix_tree.hpp"
#include "tree/karras_alg.hpp"



int main(void){


    

    mpi_init();



    

    if(world_rank == 0){
        const int dir_err = system("mkdir test_report");
        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }

    }


    


    add_test("string_utils.hpp/format()",false, []{

        UTest_NOMPI_assert(R"=(format("%s %d\n","ttt : ",10))=", ! format("%s %d\n","ttt : ",10).compare("ttt :  10\n"));

    });

    add_test("tree/morton.hpp",false, []{

        UTest_NOMPI_assert(R"=(morton::xyz_to_morton(0, 0, 0) == 0x0)=", morton::xyz_to_morton(0, 0, 0) == 0x0);

    });


    add_test("test all fail",false, []{

        UTest_NOMPI_assert("test1", false);
        UTest_NOMPI_assert("test2", false);

    });

    add_test("test partial succes",false, []{

        UTest_NOMPI_assert("test1", true);
        UTest_NOMPI_assert("test2", false);

    });

    add_test("test full succes",false, []{

        UTest_NOMPI_assert("test1", true);
        UTest_NOMPI_assert("test2", true);

    });

    add_test("test1",true, []{

        UTest_MPI_assert("test true", true);
        UTest_MPI_assert("test2 true", false);

    });


    run_tests();

    print_test_suite_report();

    mpi_close();

}