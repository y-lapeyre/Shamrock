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

#include "sys/mpi_handler.hpp"
#include "sys/sycl_handler.hpp"

#include "unittests/unit_test_handler.hpp"

#include "unittests/unittests.hpp"

#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <cstdlib>



#include "utils/string_utils.hpp"

#include "tree/morton.hpp"
#include "tree/radix_tree.hpp"
#include "tree/karras_alg.hpp"

// #include "unittests/test_global.hpp"

int main(void){


    

    mpi_init();

    init_sycl();

    

    if(world_rank == 0){
        const int dir_err = system("mkdir test_report");
        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }

    }





    run_tests();




    if(unit_test::test_start("test_1", true)){

        unit_test::test_log("test\n");

        unit_test::test_assert("test assert", true);

    }unit_test::test_end();





    
    if(world_rank == 0){
        unit_test::print_test_results();
        unit_test::write_test_results("unit_test_report.json");
    }
        
    

    /*


    


    add_tests_global();

    



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

    */

    mpi_close();

}