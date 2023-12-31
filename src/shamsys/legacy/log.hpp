// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once 

/**
 * @file log.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shambackends/typeAliasVec.hpp"
#include <chrono>
#include <fstream>
#include <string>
#include <iostream>

#include "shamsys/Log.hpp"


#include "shambase/term_colors.hpp"
#include "shamcomm/logs.hpp"



namespace logger {

    

    
    








    inline void print(){}

    template <typename T, typename... Types>
    void print(T var1, Types... var2){
        std::cout << shamcomm::logs::format_message(var1, var2...);
    }


    
 

    




    template <typename... Types>
    inline void raw(Types... var2){
        print(var2...);
    }

    template <typename... Types>
    inline void raw_ln(Types... var2){
        raw(var2...);
        std::cout << std::endl;
    }


    inline void print_faint_row(){
        raw_ln(shambase::term_colors::faint() + "-----------------------------------------------------" + shambase::term_colors::reset());
    }

    inline i8 loglevel = 0;

    #define LIST_LEVEL                                                                                                      \
    X(debug_alloc, shambase::term_colors::col8b_red(), "Debug Alloc ", 127)                                     \
    X(debug_mpi, shambase::term_colors::col8b_blue(), "Debug MPI ", 100)                                      \
    X(debug_sycl, shambase::term_colors::col8b_magenta(), "Debug SYCL", 11)                                        \
    X(debug, shambase::term_colors::col8b_green(), "Debug ", 10)                                                   \
    X(info, shambase::term_colors::col8b_cyan(), "", 1)                                                            \
    X(normal, shambase::term_colors::empty() , "", 0)                                                                \
    X(warn, shambase::term_colors::col8b_yellow(), "Warning ", -1)                                                 \
    X(err, shambase::term_colors::col8b_red(), "Error ", -10)



    //////////////////////////////
    //declare all the log levels
    //////////////////////////////
    #define DECLARE_LOG_LEVEL(_name, color, loginf, logval)                                                                     \
                                                                                                                            \
    constexpr i8 log_##_name = (logval);                                                                                              \
                                                                                                                            \
    template <typename... Types> inline void _name(std::string module_name, Types... var2) {                                \
        if (loglevel >= log_##_name) {                                                                                      \
            std::cout << "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color) + (loginf) +          \
                             shambase::term_colors::reset() + ": ";                                                                \
            logger::print(var2...);                                                                                         \
        }                                                                                                                   \
    }                                                                                                                       \
                                                                                                                            \
    template <typename... Types> inline void _name##_ln(std::string module_name, Types... var2) {                           \
        if (loglevel >= log_##_name) {                                                                                      \
            _name(module_name, var2...);                                                                                    \
            std::cout << std::endl;                                                                                         \
        }                                                                                                                   \
    }

    #define X DECLARE_LOG_LEVEL
    LIST_LEVEL
    #undef X

    #undef DECLARE_LOG_LEVEL
    ///////////////////////////////////
    // log level declared
    ///////////////////////////////////



    #define IsActivePrint(_name, color, loginf, logval) \
        if (loglevel >= log_##_name) {logger::raw("    ");} _name##_ln("xxx", "xxx","(","logger::" #_name,")");

    inline void print_active_level(){

        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
        #define X IsActivePrint
        LIST_LEVEL
        #undef X
        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);

    }

    #undef IsActivePrint








    
}




