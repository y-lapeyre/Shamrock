#pragma once

#include "../sys/mpi_handler.hpp"
#include <cstring>
#include <mpi.h>
#include <string>
#include <vector>

/***


idea :

add_test("name",true false mpi, []{

    UTest_assert("description", condition);
    UTest_assert("description", condition);

});

*/


/**
 * @brief struct to store data required to perform the test
 * @todo add (std::string test_suite_name) and then use map to group testsuite test together
 */
struct Test{
    std::string name;
    void (*test_func)();
    bool mpi_test;

    Test(std::string test_name,bool is_mpi_test ,void (*lambda_test_func)()){
        name = test_name;
        test_func = lambda_test_func;
        mpi_test = is_mpi_test;
    }
};

inline std::vector<Test> test_list;

inline void add_test(std::string name, bool mpi_test, void (*test_func)()){
    test_list.push_back(Test(name,mpi_test,test_func));
}

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
    unsigned int succes_count;
    unsigned int assert_count;
    float succes_rate;
    std::string name;
    bool is_mpi;
};



inline std::vector<Assert_report> tmp_assert_report_list; 

#define UTest_NOMPI_assert(desc,cdt) tmp_assert_report_list.push_back(Assert_report( desc, cdt))
#define UTest_MPI_assert(desc,cdt) tmp_assert_report_list.push_back(Assert_report( desc, cdt))

inline std::vector<Test_report> test_report_list;

inline void run_test_nompi(Test t){

    if(world_rank != 0) return;

    printf("runnning test : %s\n",t.name.c_str());

    tmp_assert_report_list.clear();

    t.test_func();

    Test_report t_rep;
    t_rep.name = t.name;
    t_rep.is_mpi = t.mpi_test;

    t_rep.assert_report_list = tmp_assert_report_list;
    t_rep.assert_count = t_rep.assert_report_list.size();
    
    unsigned int count_assert_succes = 0;
    for(Assert_report a : t_rep.assert_report_list){
        if(a.condition) count_assert_succes ++;
    }

    t_rep.succes_count = count_assert_succes;
    t_rep.succes_rate = float(t_rep.succes_count)/float(t_rep.assert_count);

    test_report_list.push_back(t_rep);

    tmp_assert_report_list.clear();

}

//assert can be send from whatever location
inline void run_test_mpi(Test t){

    //here the goal is to perform the parralel test and then recover the list of assert report to node 0

    if(world_rank == 0) 
        printf("runnning test : %s\n",t.name.c_str());

    tmp_assert_report_list.clear();


    t.test_func();

    //recover the assert_report from every node

    //printf("%zu\n",sizeof(Assert_report));






    //https://stackoverflow.com/questions/12080845/mpi-receive-gather-dynamic-vector-length

    int *counts = new int[world_size];
    int nelements = (int)tmp_assert_report_list.size()*sizeof(Assert_report);
    // Each process tells the root how many elements it holds
    MPI_Gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[world_size];
    // Displacement for the first chunk of data - 0
    for (int i = 0; i < world_size; i++)
        disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

    // Place to hold the gathered data
    // Allocate at root only
    char *gather_data = NULL;
    if (world_rank == 0)
        // disps[size-1]+counts[size-1] == total number of elements
        gather_data = new char[disps[world_size-1]+counts[world_size-1]];

    // Collect everything into the root
    MPI_Gatherv(&tmp_assert_report_list[0], nelements, MPI_CHAR,
                gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0){
        Assert_report* rep_dat = (Assert_report*) gather_data;
        unsigned int len_rep_dat = (disps[world_size-1]+counts[world_size-1])/sizeof(Assert_report);

        //printf("len_rep_dat : %d\n",len_rep_dat);

        Test_report t_rep;
        t_rep.name = t.name;
        t_rep.is_mpi = t.mpi_test;

        for(unsigned int i = 0; i < len_rep_dat; i++){
            t_rep.assert_report_list.push_back(rep_dat[i]);
        }

        t_rep.assert_count = t_rep.assert_report_list.size();
        
        unsigned int count_assert_succes = 0;
        for(Assert_report a : t_rep.assert_report_list){
            if(a.condition) count_assert_succes ++;
        }

        t_rep.succes_count = count_assert_succes;
        t_rep.succes_rate = float(t_rep.succes_count)/float(t_rep.assert_count);

        test_report_list.push_back(t_rep);
    }




   

}

/**
 * @brief run all tests
 * @todo add timer for each test
 */
inline void run_tests(){

    for(Test t : test_list){

        mpi_barrier();

        if(t.mpi_test){
            run_test_mpi(t);
        }else{
            run_test_nompi(t);
        }

    }

    mpi_barrier();

}

/**
 * @todo move defines to aliases.hpp
 * 
 */
#define STR_(X) #X
#define STR(X) STR_(X)

#define COLOR_NO "\033[0m"
#define COLOR_RED "\x1B[31m"
#define COLOR_GREEN "\x1B[32m"
#define COLOR_YELLOW "\x1B[33m"
#define COLOR_BRIGHT_RED "\x1B[91m"
#define COLOR_BRIGHT_BLUE "\x1B[94m"

inline void print_test_suite_report(){

    if(world_rank != 0) return;


    printf("printing test report: \n");
    printf("--------------------------------------------\n");

    for(Test_report trep : test_report_list){

        if(trep.succes_rate == 0){
            printf("-> " COLOR_BRIGHT_BLUE "%-32s" COLOR_NO " | " COLOR_BRIGHT_RED "%5.1f" COLOR_NO " %% (" COLOR_BRIGHT_RED "%d" COLOR_NO "/" COLOR_BRIGHT_RED "%d" COLOR_NO ")\n",trep.name.c_str(), 100*trep.succes_rate, trep.succes_count, trep.assert_count);
        }else if (trep.succes_rate > 0 && trep.succes_rate < 1) {
            printf("-> " COLOR_BRIGHT_BLUE "%-32s" COLOR_NO " | " COLOR_YELLOW     "%5.1f" COLOR_NO " %% (" COLOR_YELLOW "%d" COLOR_NO "/" COLOR_YELLOW "%d" COLOR_NO ")\n",trep.name.c_str(), 100*trep.succes_rate, trep.succes_count, trep.assert_count);
        }else if (trep.succes_rate >= 1){
            printf("-> " COLOR_BRIGHT_BLUE "%-32s" COLOR_NO " | " COLOR_GREEN      "%5.1f" COLOR_NO " %% (" COLOR_GREEN "%d" COLOR_NO "/" COLOR_GREEN "%d" COLOR_NO ")\n",trep.name.c_str(), 100*trep.succes_rate, trep.succes_count, trep.assert_count);
        }

        
        for(Assert_report arep : trep.assert_report_list){

            if(trep.is_mpi){
                if(arep.condition){
                    printf("     |  " COLOR_GREEN "%-" STR(DECRIPTION_ASSERT_MAX_LEN) "s" COLOR_NO " [%3d]\n",arep.description,arep.rank_emited);
                }else{
                    printf("     |  " COLOR_RED   "%-" STR(DECRIPTION_ASSERT_MAX_LEN) "s" COLOR_NO " [%3d]\n",arep.description,arep.rank_emited);
                }
            }else{
                if(arep.condition){
                    printf("     |  " COLOR_GREEN "%-" STR(DECRIPTION_ASSERT_MAX_LEN) "s" COLOR_NO " \n",arep.description);
                }else{
                    printf("     |  " COLOR_RED   "%-" STR(DECRIPTION_ASSERT_MAX_LEN) "s" COLOR_NO " \n",arep.description);
                }
            }
        }
    }
}

