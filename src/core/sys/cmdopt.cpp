// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "cmdopt.hpp"
#include "core/sys/log.hpp"
#include "runscript/shamrockapi.hpp"
#include <optional>
#include <string_view>
#include <vector>
#include <string>

namespace opts {

    std::string_view executable_name;
    std::vector<std::string_view> args;
    bool init_done;




    struct Opts{
        std::string name;
        std::optional<std::string> args;
        std::string description;
    };

    std::vector<Opts> registered_opts;

    bool is_name_registered(const std::string_view & name){
        for (auto opt : registered_opts) {
            if(opt.name == name){
                return true;
            }
        }
        return false;
    }




    void check_args_registered(){
        bool error = false;

        std::string err_buf;

        for (auto arg : args) {
            if(arg.rfind("-",0)==0){
                if(!is_name_registered(arg)){
                    logger::err_ln("opts", "argument :",arg, "is not registered");
                    err_buf += "\"";
                    err_buf += arg;
                    err_buf += "\"";
                    err_buf += " ";
                    error = true;
                }
            }
        }

        if(error){
            throw ShamAPIException(err_buf + "names are not registered in ::opts");
        }
    }




    void check_init(){
        if(! init_done) throw ShamAPIException("Cmdopt uninitialized");
    }



    

    bool has_option(const std::string_view &option_name) {
        check_init();

        if(!is_name_registered(option_name)){
            logger::err_ln("opts", "argument :",option_name, "is not registered");
             throw ShamAPIException(std::string(option_name) + " option is not registered in ::opts");
        }

        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                return true;
        }

        return false;
    }

    std::string_view get_option(const std::string_view &option_name) {
        check_init();

        if(!is_name_registered(option_name)){
            logger::err_ln("opts", "argument :",option_name, "is not registered");
             throw ShamAPIException(std::string(option_name) + " option is not registered in ::opts");
        }
        
        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                if (it + 1 != end)
                    return *(it + 1);
        }

        return "";
    }


    void register_opt(std::string name,std::optional<std::string> args ,std::string description){
        
        
        registered_opts.push_back({name,args,description});
    }

    


    


    void init(int argc, char *argv[]) { 
        opts::register_opt("--help",{}, "show this message");
        executable_name = std::string_view(argv[0]);
        args = std::vector<std::string_view>(argv + 1, argv + argc); 
        init_done = true;
        check_args_registered();
    }


    void print_help(){
        logger::raw_ln("executable :",executable_name.data());

        logger::raw_ln("\nUsage :");

        for (auto & [n,arg,desc] : registered_opts) {

            std::string arg_print = arg.value_or("");
            
            logger::raw_ln(format("%-15s %-15s",n.c_str(),arg_print.c_str())," :",desc);
            
        }

    }

    bool is_help_mode(){
        if(has_option("--help")){
            print_help();return true;
        }else{
            return false;
        }
    }



}