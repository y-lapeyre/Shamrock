/**
 * @file cmdopt.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * @version 0.1
 * @date 2022-03-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "aliases.hpp"

class Cmdopt {
  public:
  private:
    Cmdopt(){};
    Cmdopt(const Cmdopt &);
    Cmdopt &operator=(const Cmdopt &);

    bool init_done = false;
    std::vector<std::string_view> args;

    std::string_view executable_name;
    std::string usage_str;

  public:

    inline void check_init(){
        if(! init_done) throw shamrock_exc("Cmdopt uninitialized");
    }
    

    bool has_option(const std::string_view &option_name) {
        check_init();
        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                return true;
        }

        return false;
    }

    std::string_view get_option(const std::string_view &option_name) {
        check_init();
        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                if (it + 1 != end)
                    return *(it + 1);
        }

        return "";
    }



    /**
     * @brief init sycl handler
     *
     */
    inline void init(int argc, char *argv[],std::string usage_string) { 
        executable_name = std::string_view(argv[0]);
        args = std::vector<std::string_view>(argv + 1, argv + argc); 
        init_done = true;
        usage_str = usage_string;
    }

    inline void print_help(){
        printf("executable : %s",executable_name.data());
        printf("%s",usage_str.c_str());
    }

    inline bool is_help_mode(){
        if(has_option("--help")){
            print_help();return true;
        }else{
            return false;
        }
    }

    

    /**
     * @brief Get the unique instance of the cmdopt handler
     *
     * @return Cmdopt& sycl handler instance
     */
    static Cmdopt &get_instance() {
        static Cmdopt instance;
        return instance;
    }


};