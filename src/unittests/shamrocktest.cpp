// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good



#include "unittests/shamrocktest.hpp"
#include <cstdlib>
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

    


    using namespace mpi_handler;
    


    


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



    mpi_handler::init(argc,argv);printf("\n");

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


std::string matplotlibstyle = R"==(

# Set color cycle: blue, green, yellow, red, violet, gray
axes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])

# Set default figure size
figure.figsize : 10,5

# Set x axis
xtick.direction : in
xtick.major.size : 5
xtick.major.width : 1
xtick.minor.size : 3
xtick.minor.width : 1
xtick.minor.visible : True
xtick.top : True

# Set y axis
ytick.direction : in
ytick.major.size : 5
ytick.major.width : 1
ytick.minor.size : 3
ytick.minor.width : 1
ytick.minor.visible : True
ytick.right : True

# Set line widths
axes.linewidth : 1.5
grid.linewidth : 0.5
lines.linewidth : 1.

# Remove legend frame
legend.frameon : False

# Always save as 'tight'
savefig.bbox : tight
savefig.pad_inches : 0.05

# Use serif fonts
# font.serif : Times
font.family : serif
font.size : 15
mathtext.fontset : dejavuserif

# Use LaTeX for math formatting
text.usetex : True
text.latex.preamble : \usepackage{amsmath} \usepackage{amssymb}

)==";




void run_py_script(std::string pysrc){

    auto style_exist = []() -> bool {
        std::ifstream f("custom_style.mplstyle");
        return f.good();
    };

    if (!style_exist()) {
        std::ofstream style_file("custom_style.mplstyle");
        style_file << matplotlibstyle;
        style_file.close();
    }

    std::ofstream py_file("tmp.py");
    py_file << pysrc;
    py_file.close();

    system("python3 tmp.py");
}





namespace impl::shamrocktest {

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
            acc += format( "%e" , vec[i]) ;
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
                R"(    "compute_queue" : ")" + sycl_handler::get_compute_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                R"(    "alt_queue" : ")" + sycl_handler::get_alt_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                R"(    "world_rank" : )" + std::to_string(world_rank) + ",\n" +
                R"(    "asserts" : )" + increase_indent( asserts.serialize())+ ",\n" +
                R"(    "test_data" : )" + increase_indent( test_data.serialize()) + "\n" +
                "}";
        };


        return get_str();
    }



}

namespace shamrock::test {
    int run_all_tests(int argc, char *argv[], bool run_bench,bool run_analysis, bool run_unittest){
        
        
        using namespace mpi_handler;
    
        using namespace impl::shamrocktest;



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



        mpi_handler::init(argc,argv);printf("\n");

        sycl_handler::init();

        if(!run_only){
            printf("\n------------ Tests list --------------\n");
        }

        std::vector<u32> selected_tests = {};

        for (u32 i = 0; i < static_init_vec_tests.size(); i++) {
            
            if(run_only){
                if(run_only_name.compare(static_init_vec_tests[i].name) ==0){
                    selected_tests.push_back(i);
                }
            }else{
                bool any_node_cnt = (static_init_vec_tests[i].node_count == -1);
                if(any_node_cnt || (static_init_vec_tests[i].node_count == world_size)){
                    selected_tests.push_back(i);
                    if(any_node_cnt){
                        printf(" - [\033[;32many\033[0m] ");
                    }else{
                        printf(" - [\033[;32m%03d\033[0m] ",static_init_vec_tests[i].node_count);
                    }
                    
                    std::cout << "\033[;32m"<<static_init_vec_tests[i].name <<  "\033[0m " <<std::endl;
                }else{
                    printf(" - [\033[;31m%03d\033[0m] ",static_init_vec_tests[i].node_count);
                    std::cout << "\033[;31m"<<static_init_vec_tests[i].name <<  "\033[0m " <<std::endl;
                }
            }

        }

        if(!run_only){
            printf("--------------------------------------\n\n");
        }



        std::stringstream rank_test_res_out;




        u32 test_loc_cnt = 0;

        for (u32 i : selected_tests) {

            impl::shamrocktest::Test & test = static_init_vec_tests[i];

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

            std::string s_assert = format(" [%d/%d] ",succes_cnt,res.asserts.asserts.size());
            printf("%-15s",s_assert.c_str());
            std::cout << " (" << timer.get_time_str() << ")" <<std::endl;

            
            std::cout << std::endl;

            rank_test_res_out << res.serialize() << ",";


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
        

        //generate json output and write it into the specified file
        if(world_rank == 0){

            if(out_res_string.back() == ','){
                out_res_string = out_res_string.substr(0,out_res_string.size()-1);
            }

            std::string s_out;

            s_out = "{\n";

            s_out += R"(    "commit_hash" : ")" + git_commit_hash + "\",\n" ;
            s_out += R"(    "world_size" : ")" + std::to_string(mpi_handler::world_size) + "\",\n" ;
            s_out += R"(    "results" : )"   "[\n\n" ;
            s_out += increase_indent(out_res_string);
            s_out += "\n    ]\n}";

            //printf("%s\n",s_out.c_str());

            if(out_to_file){
                write_string_to_file(std::string(opts::get_option("-o")), s_out);
            }
            
        }


        mpi_handler::close();


        return 0;
    }
}