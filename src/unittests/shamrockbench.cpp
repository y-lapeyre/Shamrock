// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "unittests/shamrockbench.hpp"
#include <cmath>
#include <sstream>

#include "core/sys/cmdopt.hpp"
#include "core/utils/string_utils.hpp"
#include "core/sys/sycl_handler.hpp"


std::string make_test_output(BenchmarkResults t_res){

    std::stringstream output;

    output << "%bench_name = \"" << t_res.bench_name << "\"" << std::endl;

    output << "%world_size = " << mpi_handler::world_size << std::endl;
    output << "%world_rank = " << mpi_handler::world_rank << std::endl;

    for(auto s : t_res.scores){
        output << "%result = " << s <<std::endl;
    }

    output << "%end_bench" << std::endl;

    return output.str();
}


//TODO add memory clean as an assertion
int run_all_bench(int argc, char *argv[]){

    //std::cout << shamrock_title_bar_big << std::endl;


    using namespace mpi_handler;
    


    if(opts::is_help_mode()){
        return 0;
    }

    if(opts::has_option("--bench-list")){
        for (auto b : benchmarks) {
            if(b.node_count == -1){
                printf("- [any] %-15s (%s)\n",b.title.c_str(),b.tech_name.c_str());
            }else{
                printf("- [%03d] %-15s (%s)\n",b.node_count,b.title.c_str(),b.tech_name.c_str());
            }
        }
        return 0;
    }

    bool run_only = false;
    std::string run_only_name = "";
    if(opts::has_option("--run-only")){
        run_only_name = opts::get_option("--run-only");
        run_only = true;
    }


    bool full_output = opts::has_option("--full-output");

    bool out_to_file = opts::has_option("-o");

    if(out_to_file){
        if(opts::get_option("-o").size() == 0){
            opts::print_help();
        }
    }



    mpi_handler::init(argc,argv);printf("\n");

    sycl_handler::init();

    if(!run_only){
        printf("\n------------ Tests list --------------\n");
    }

    std::vector<BenchmarkEntry> selected_tests = {};

    for (auto b : benchmarks) {
        
        if(run_only){
            if(run_only_name.compare(b.tech_name) ==0){
                selected_tests.push_back(b);
            }
        }else{
            bool any_node_cnt = (b.node_count == -1);
            if(any_node_cnt || (b.node_count == world_size)){
                selected_tests.push_back(b);
                if(any_node_cnt){
                    printf(" - [\033[;32many\033[0m] ");
                }else{
                    printf(" - [\033[;32m%03d\033[0m] ",b.node_count);
                }
                
                std::cout << "\033[;32m"<<b.title <<  "\033[0m " <<std::endl;
            }else{
                printf(" - [\033[;31m%03d\033[0m] ",b.node_count);
                std::cout << "\033[;31m"<<b.title <<  "\033[0m " <<std::endl;
            }
        }

    }

    if(!run_only){
        printf("--------------------------------------\n\n");
    }



    std::stringstream rank_test_res_out;




    u32 test_loc_cnt = 0;

    for (auto b : selected_tests) {

        if(run_only){
            printf("running test : ");
        }else{
            printf("running test [%d/%zu] : ",test_loc_cnt+1,selected_tests.size());
        }

        bool any_node_cnt = b.node_count == -1;
        if(any_node_cnt){
            printf(" [any] ");
        }else{
            printf(" [%03d] ",b.node_count);
        }

        std::cout << "\033[;34m"<<b.title <<  "\033[0m " <<std::endl;
        


        BenchmarkResults t(b.title);


        #ifdef MEM_TRACK_ENABLED
        u64 alloc_cnt_before_test = ptr_allocated.size();
        #endif
        mpi::barrier(MPI_COMM_WORLD);
        Timer timer;
        timer.start();
        b.func(t);
        timer.end();
        mpi::barrier(MPI_COMM_WORLD);

        
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
        
        u32 cnt = t.scores.size();
        f64 avg = 0.f;

        for (f64 s : t.scores){
            avg += s/cnt;
        }

        f64 var = 0.f;
        for (f64 s : t.scores){
            var += (s - avg)*(s - avg)/cnt;
        }

        f64 sigma = std::sqrt(var);


        std::string identifier = "ns";
        f64 div = 1;

        if (avg > 1000){
            identifier = "mus";
            div *= 1000;
        }

        if (avg/div > 1000){
            identifier = "ms";
            div *= 1000;
        }

        if (avg/div > 1000){
            identifier = "s";
            div *= 1000;
        }
            
        std::cout << "       Result : avg = " << avg/div <<" "<< identifier <<" sigma = " << sigma/div <<" "<< identifier <<std::endl;
        
        std::cout << " (" << timer.get_time_str() << ")" <<std::endl;

        
        std::cout << std::endl;

        rank_test_res_out << make_test_output(t) << std::endl << std::endl;


        test_loc_cnt++;

    }



    //recover result on node 0 and write to file if -o <outfile> specified
    std::string out_res_string;

    if(world_size == 1){
        out_res_string = rank_test_res_out.str();
    }else{
        std::string loc_string = rank_test_res_out.str();


        //printf("sending : \n%s\n",loc_string.c_str());

        int *counts = new int[world_size];
        int nelements = (int) loc_string.size();
        // Each process tells the root how many elements it holds
        mpi::gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);


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
        mpi::gatherv(loc_string.c_str(), nelements, MPI_CHAR,
                    gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

        
        if(world_rank == 0){
            out_res_string = std::string(gather_data,disps[world_size-1]+counts[world_size-1]);
        }

    }
    


    if(world_rank == 0){

        std::string s_out = out_res_string;

        //printf("%s\n",s_out.c_str());

        if(out_to_file){
            write_string_to_file(std::string(opts::get_option("-o")), s_out);
        }
        
    }


    mpi_handler::close();


    return 0;
}