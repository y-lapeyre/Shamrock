// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "core/sys/cmdopt.hpp"
#include "core/sys/log.hpp"
#include "unittests/shamrockbench.hpp"
#include "unittests/shamrocktest.hpp"

int main(int argc, char *argv[]){


    std::cout << shamrock_title_bar_big << std::endl;

    opts::register_opt("--sycl-cfg","(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--loglevel","(logvalue)", "specify a log level");
    opts::register_opt("--nocolor",{}, "disable colored ouput");

    opts::register_opt("--test-list",{}, "print test availables");
    opts::register_opt("--bench-list",{}, "print test availables");
    opts::register_opt("--run-only",{"(test name)"}, "run only this test");
    opts::register_opt("--full-output",{}, "print the assertions in the tests");

    opts::register_opt("--benchmark",{}, "print the assertions in the tests");


    opts::register_opt("-o",{"(filepath)"}, "output test report in that file");


    opts::init(argc, argv);
    if(opts::is_help_mode()){
        return 0;
    }

    if(opts::has_option("--loglevel")){
        std::string level = std::string(opts::get_option("--loglevel"));

        i32 a = atoi(level.c_str());

        if(i8(a) != a){
            logger::err_ln("Cmd OPT", "you must select a loglevel in a 8bit integer range");
        }

        logger::loglevel = a;

        if(a == i8_max){
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln("-> modified loglevel to",logger::loglevel,"enabled log types : ");
        logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
        logger::print_active_level();
        logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
    }

    if(opts::has_option("--benchmark")){
        return run_all_bench(argc,argv);
    }else{
        return run_all_tests(argc,argv);
    }
}