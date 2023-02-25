// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good



#include "shamtest.hpp"
#include <cstdlib>
#include <sstream>

#include "shamrock/legacy/utils/time_utils.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamrock/legacy/utils/string_utils.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

#include "shamsys/legacy/mpi_handler.hpp"


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



namespace shamtest::details {

    std::string TestAssert::serialize(){
        std::string acc = "\n{\n";

        acc += R"(    "value" : )" + std::to_string(value) + ",\n" ;
        acc += R"(    "name" : ")" + name + "\"" ;

        if(! comment.empty()){
            acc += ",\n" R"(    "comment" : ")" + comment + "\"" ;
        }

        acc += "\n}";
        return acc;
    }

    std::string serialize_vec(const std::vector<f64> & vec){
        std::string acc = "\n[\n";

        for(u32 i = 0; i < vec.size(); i++){
            acc += shamutils::format_printf( "%e" , vec[i]) ;
            if(i < vec.size()-1){
                acc += ", ";
            }
        }

        acc += "\n]";
        return acc;
    }

    std::string DataNode::serialize(){
        std::string acc = "\n{\n";

        acc += R"(    "name" : ")" + name + "\",\n" ;

        acc += R"(    "data" : )" "\n"+ serialize_vec(data) + "\n" ;

        acc += "\n}";
        return acc;
    }

    std::string TestData::serialize(){
        std::string acc = "\n{\n";

        acc += R"(    "dataset_name" : ")" + dataset_name + "\",\n" ;
        acc += R"(    "dataset" : )" "\n    [\n" ;

        for(u32 i = 0; i < dataset.size(); i++){
            acc += increase_indent( dataset[i].serialize()) ;
            if(i < dataset.size()-1){
                acc += ",";
            }
        }

        acc += "]" ;

        acc += "\n}";
        return acc;
    }

    std::string TestAssertList::serialize(){
        std::string acc = "\n[\n";

        for(u32 i = 0; i < asserts.size(); i++){
            acc += increase_indent( asserts[i].serialize()) ;
            if(i < asserts.size()-1){
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }

    std::string TestDataList::serialize(){
        std::string acc = "\n[\n";

        for(u32 i = 0; i < test_data.size(); i++){
            acc += increase_indent( test_data[i].serialize()) ;
            if(i < test_data.size()-1){
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }


    std::string TestResult::serialize(){

        using namespace shamsys::instance;

        auto get_type_name = [](TestType t) -> std::string {
            switch (t) {
                case Benchmark: return "Benchmark";
                case Analysis:  return "Analysis";
                case Unittest:  return "Unittest";
            }
        };

        auto get_str = [&]() -> std::string {
            return 
                "{\n"
                R"(    "type" : ")" + get_type_name(type) + "\",\n" +
                R"(    "name" : ")" + name + "\",\n" +
                R"(    "compute_queue" : ")" + get_compute_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                R"(    "alt_queue" : ")" + get_alt_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                R"(    "world_rank" : )" + std::to_string(world_rank) + ",\n" +
                R"(    "asserts" : )" + increase_indent( asserts.serialize())+ ",\n" +
                R"(    "test_data" : )" + increase_indent( test_data.serialize()) + "\n" +
                "}";
        };


        return get_str();
    }

    TestResult current_test{Unittest,"",-1};


    TestResult Test::run(){

        using namespace shamsys::instance;

        if(node_count != -1){
            if(node_count != world_size){
                throw excep_with_pos(std::runtime_error,"trying to run a test with wrong number of nodes");
            }
        }

        current_test = TestResult{type, name,world_rank};

        test_functor();


        return std::move(current_test);
    }



}

namespace shamtest {
    int run_all_tests(int argc, char *argv[], bool run_bench,bool run_analysis, bool run_unittest){
        
        
        using namespace mpi_handler;
    
        using namespace shamtest::details;



        if(opts::has_option("--test-list")){


            auto print_list = [&](TestType t){
                for (auto test : static_init_vec_tests) {
                    if (test.type == t) {
                        if(test.node_count == -1){
                            printf("- [any] %-15s\n",test.name.c_str());
                        }else{
                            printf("- [%03d] %-15s\n",test.node_count,test.name.c_str());
                        }
                    }
                }
            };

            printf("--- Benchmark ---\n");

            print_list(Benchmark);

            printf("--- Analysis  ---\n");

            print_list(Analysis);

            printf("--- Unittest  ---\n");

            print_list(Unittest);

            
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


        using namespace shamsys;

        instance::init(argc,argv);printf("\n");

        if(!run_only){
            printf("\n------------ Tests list --------------\n");
        }

        std::vector<u32> selected_tests = {};

        for (u32 i = 0; i < static_init_vec_tests.size(); i++) {


            auto can_run = [&](shamtest::details::Test & t) -> bool {

                bool any_node_cnt = (t.node_count == -1);
                bool world_size_ok = t.node_count == instance::world_size;

                bool can_run_type = false;

                auto test_type = t.type;
                can_run_type = can_run_type || (run_unittest && (Unittest == test_type)) ;
                can_run_type = can_run_type || (run_analysis && (Analysis == test_type)) ;
                can_run_type = can_run_type || (run_bench && (Benchmark == test_type)) ;

                return can_run_type && (any_node_cnt || world_size_ok );
            };

            auto print_test = [&](shamtest::details::Test & t, bool enabled){
                bool any_node_cnt = (t.node_count == -1);

                if (enabled) {
                
                    if(any_node_cnt){
                        printf(" - [\033[;32many\033[0m] ");
                    }else{
                        printf(" - [\033[;32m%03d\033[0m] ",static_init_vec_tests[i].node_count);
                    }
                    std::cout << "\033[;32m"<<static_init_vec_tests[i].name <<  "\033[0m " <<std::endl;

                }else{
                    if(any_node_cnt){
                        printf(" - [\033[;31many\033[0m] ");
                    }else{
                        printf(" - [\033[;31m%03d\033[0m] ",static_init_vec_tests[i].node_count);
                    }
                    std::cout << "\033[;31m"<<static_init_vec_tests[i].name <<  "\033[0m " <<std::endl;
                }


            };

            bool run_test = can_run(static_init_vec_tests[i]);

            if(run_only){
                if(run_only_name.compare(static_init_vec_tests[i].name) ==0){
                    if(run_test) {
                        selected_tests.push_back(i);
                    }else{
                        logger::err_ln("TEST", "test can not run under the following configuration");
                    }
                }
            }else{
                print_test(static_init_vec_tests[i], run_test);
                if(run_test) {selected_tests.push_back(i);}
            }

        }

        if(!run_only){
            printf("--------------------------------------\n\n");
        }



        std::stringstream rank_test_res_out;




        u32 test_loc_cnt = 0;

        for (u32 i : selected_tests) {

            shamtest::details::Test & test = static_init_vec_tests[i];

            if(run_only){
                printf("running test : ");
            }else{
                printf("running test [%d/%zu] : ",test_loc_cnt+1,selected_tests.size());
            }

            bool any_node_cnt = test.node_count == -1;
            if(any_node_cnt){
                printf(" [any] ");
            }else{
                printf(" [%03d] ",test.node_count);
            }

            std::cout << "\033[;34m"<<test.name <<  "\033[0m " <<std::endl;
            




            mpi::barrier(MPI_COMM_WORLD);
            Timer timer;
            timer.start();
            auto res = test.run();
            timer.end();
            mpi::barrier(MPI_COMM_WORLD);

            
            
            for(unsigned int j = 0; j < res.asserts.asserts.size(); j++){


                if(full_output || (!res.asserts.asserts[j].value)){
                    printf("        [%d/%zu] : ",j+1,res.asserts.asserts.size());
                    printf("%-20s",res.asserts.asserts[j].name.c_str());
                    
                    if(res.asserts.asserts[j].value){
                        std::cout << "  (\033[;32mSucces\033[0m)\n";
                    }else{
                        std::cout << "  (\033[1;31m Fail \033[0m)\n";
                        if(! res.asserts.asserts[j].comment.empty()){
                            std::cout << "----- logs : \n" << res.asserts.asserts[j].comment << "\n-----" << std::endl;
                        }
                    }
                }
            }

                
            

            u32 succes_cnt = 0;
            for(unsigned int j = 0; j < res.asserts.asserts.size(); j++){
                if(res.asserts.asserts[j].value){
                    succes_cnt++;
                }
            }
            
            if(succes_cnt == res.asserts.asserts.size()){
                std::cout << "       Result : \033[;32mSucces\033[0m";
            }else{
                std::cout << "       Result : \033[1;31m Fail \033[0m";
            }



            std::string s_assert = shamsys::format(" [{}/{}] ",succes_cnt,res.asserts.asserts.size());
            printf("%-15s",s_assert.c_str());
            std::cout << " (" << timer.get_time_str() << ")" <<std::endl;

            
            std::cout << std::endl;

            rank_test_res_out << res.serialize() << ",";


            test_loc_cnt++;

        }



        //recover result on node 0 and write to file if -o <outfile> specified
        std::string out_res_string;

        if(instance::world_size == 1){
            out_res_string = rank_test_res_out.str();
        }else{
            std::string loc_string = rank_test_res_out.str();


            //printf("sending : \n%s\n",loc_string.c_str());

            int *counts = new int[instance::world_size];
            int nelements = (int) loc_string.size();
            // Each process tells the root how many elements it holds
            mpi::gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);


            // Displacements in the receive buffer for MPI_GATHERV
            int *disps = new int[instance::world_size];
            // Displacement for the first chunk of data - 0
            for (int i = 0; i < instance::world_size; i++)
                disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

            // Place to hold the gathered data
            // Allocate at root only
            char *gather_data = NULL;
            if (instance::world_rank == 0)
                // disps[size-1]+counts[size-1] == total number of elements
                gather_data = new char[disps[instance::world_size-1]+counts[instance::world_size-1]];


            // Collect everything into the root
            mpi::gatherv(loc_string.c_str(), nelements, MPI_CHAR,
                        gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

            
            if(instance::world_rank == 0){
                out_res_string = std::string(gather_data,disps[instance::world_size-1]+counts[instance::world_size-1]);
            }

        }
        

        //generate json output and write it into the specified file
        if(instance::world_rank == 0){

            if(out_res_string.back() == ','){
                out_res_string = out_res_string.substr(0,out_res_string.size()-1);
            }

            std::string s_out;

            s_out = "{\n";

            s_out += R"(    "commit_hash" : ")" + git_commit_hash + "\",\n" ;
            s_out += R"(    "world_size" : ")" + std::to_string(instance::world_size) + "\",\n" ;

            #if defined (SYCL_COMP_DPCPP)
            s_out += R"(    "compiler" : "DPCPP",)" "\n" ;
            #elif defined (SYCL_COMP_HIPSYCL)
            s_out += R"(    "compiler" : "HipSYCL",)" "\n" ;
            #else
            s_out += R"(    "compiler" : "Unknown",)" "\n" ;
            #endif  

            s_out += R"(    "comp_args" : ")" + compile_arg + "\",\n" ;


            s_out += R"(    "results" : )"   "[\n\n" ;
            s_out += increase_indent(out_res_string);
            s_out += "\n    ]\n}";

            //printf("%s\n",s_out.c_str());

            if(out_to_file){
                write_string_to_file(std::string(opts::get_option("-o")), s_out);
            }
            
        }


        instance::close();


        return 0;
    }
}