// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "unittests/shamrocktest.hpp"
#include <sstream>

#include "core/sys/cmdopt.hpp"
#include "core/utils/string_utils.hpp"
#include "core/sys/sycl_handler.hpp"


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



std::string make_test_output(TestResults t_res){

    std::stringstream output;

    output << "%test_name = \"" << t_res.test_name << "\"" << std::endl;

    output << "%world_size = " << mpi_handler::world_size << std::endl;
    output << "%world_rank = " << mpi_handler::world_rank << std::endl;

    for(unsigned int j = 0; j < t_res.lst_assert.size(); j++){
        output << "%start_assert \"" << t_res.lst_assert[j].assert_name << "\""<<std::endl;
        output << "%result = " << t_res.lst_assert[j].success <<std::endl;

        if(!t_res.lst_assert[j].log.empty()){
             output << "%startlog" <<std::endl << t_res.lst_assert[j].log << std::endl <<  "%endlog" << std::endl;
        }

        output << "%end_assert" <<std::endl;
    }

    output << "%end_test" << std::endl;

    return output.str();
}


//TODO add memory clean as an assertion
int run_all_tests(int argc, char *argv[]){

    std::cout << shamrock_title_bar_big << std::endl;

    opts::register_opt("--sycl-cfg","(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--loglevel","(logvalue)", "specify a log level");
    opts::register_opt("--nocolor",{}, "disable colored ouput");

    opts::register_opt("--test-list",{}, "print test availables");
    opts::register_opt("--run-only",{"(test name)"}, "run only this test");
    opts::register_opt("--full-output",{}, "print the assertions in the tests");


    opts::register_opt("-o",{"(filepath)"}, "output test report in that file");


    opts::init(argc, argv);

    using namespace mpi_handler;
    


    if(opts::is_help_mode()){
        return 0;
    }

    if(opts::has_option("--test-list")){
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



    mpi_handler::init();printf("\n");

    sycl_handler::init();

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



    std::stringstream rank_test_res_out;




    u32 test_loc_cnt = 0;

    for (u32 i : selected_tests) {

        if(run_only){
            printf("running test : ");
        }else{
            printf("running test [%d/%zu] : ",test_loc_cnt+1,selected_tests.size());
        }

        bool any_node_cnt = test_node_count[i] == -1;
        if(any_node_cnt){
            printf(" [any] ");
        }else{
            printf(" [%03d] ",test_node_count[i]);
        }

        std::cout << "\033[;34m"<<test_name_lst[i] <<  "\033[0m " <<std::endl;
        


        TestResults t(test_name_lst[i]);


        #ifdef MEM_TRACK_ENABLED
        u64 alloc_cnt_before_test = ptr_allocated.size();
        #endif
        mpi::barrier(MPI_COMM_WORLD);
        Timer timer;
        timer.start();
        test_fct_lst[i](t);
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
        
        for(unsigned int j = 0; j < t.lst_assert.size(); j++){


            if(full_output || (!t.lst_assert[j].success)){
                printf("        [%d/%zu] : ",j+1,t.lst_assert.size());
                printf("%-20s",t.lst_assert[j].assert_name.c_str());
                
                if(t.lst_assert[j].success){
                    std::cout << "  (\033[;32mSucces\033[0m)\n";
                }else{
                    std::cout << "  (\033[1;31m Fail \033[0m)\n";
                    std::cout << "----- logs : \n" << t.lst_assert[j].log << "\n-----" << std::endl;
                }
            }
            
            //std::cout << "            logs : " << t.lst_assert[j].log << "\n";
        }

            
        

        u32 succes_cnt = 0;
        for(unsigned int j = 0; j < t.lst_assert.size(); j++){
            if(t.lst_assert[j].success){
                succes_cnt++;
            }
        }
        
        if(succes_cnt == t.lst_assert.size()){
            std::cout << "       Result : \033[;32mSucces\033[0m";
        }else{
            std::cout << "       Result : \033[1;31m Fail \033[0m";
        }

        std::string s_assert = format(" [%d/%d] ",succes_cnt,t.lst_assert.size());
        printf("%-15s",s_assert.c_str());
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