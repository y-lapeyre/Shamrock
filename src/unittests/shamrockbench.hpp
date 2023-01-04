// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "shamsys/mpi_handler.hpp"
#include "shamsys/sycl_handler.hpp"
#include "aliases.hpp"
#include "core/utils/time_utils.hpp"



//%Impl status : Clean

class BenchmarkResults{public:
    std::string bench_name;


    std::vector<std::string> entries = std::vector<std::string>();

    BenchmarkResults(std::string parbench_name){
        bench_name = parbench_name;
    }
};

struct BenchmarkEntry{
    int node_count;
    std::string title;
    std::string tech_name;
    void (*func)(BenchmarkResults &);
};

inline std::vector<BenchmarkEntry> benchmarks;


class BenchmarkRegister{public:
    BenchmarkRegister(BenchmarkEntry b){
        benchmarks.push_back(b);
    }
}; 



#define Bench_start(tech_name,title,func_name, node_cnt) void bench_func_##func_name (BenchmarkResults& t);\
void (*bench_func_ptr_##func_name)(BenchmarkResults&) = bench_func_##func_name;\
BenchmarkRegister bench_class_obj_##func_name (BenchmarkEntry{node_cnt, title, tech_name, bench_func_ptr_##func_name});\
void bench_func_##func_name (BenchmarkResults& __bench_result_ref)



#define Register_score(result)  \
{\
    __bench_result_ref.entries.push_back(result);\
}



//#define Timeit(code) \
//{Timer timer; timer.start(); \
//{ \
//    code ; \
//} \
//timer.end(); __bench_result_ref.scores.push_back(timer.nanosec);}
//
//
//
//#define TimeitFor(count,code) \
//for(u32 __cnt = 0 ; __cnt < count ; __cnt ++){Timer timer; timer.start(); \
//{ \
//    code ; \
//} \
//timer.end(); __bench_result_ref.scores.push_back(timer.nanosec);}

int run_all_bench(int argc, char *argv[]);