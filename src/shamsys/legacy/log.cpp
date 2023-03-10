// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "log.hpp"
#include "shambase/stringUtils.hpp"
#include <chrono>

template<typename T>
u64 get_now_val(){
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ms = std::chrono::time_point_cast<T>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<T>(epoch);
    
    return value.count();
}

namespace logfiles {


    bool logfiles_on = false;


    namespace timings {
        std::ofstream timings_file;

        void register_begin(std::string name){
            auto now_val = get_now_val<std::chrono::microseconds>();

            const char* rstr =  
R"({
    "cat": "%s",  
    "pid": %d,  
    "tid": %d, 
    "ts": %zu, 
    "ph": "B", 
    "name": "%s", 
    "args": { 
    }
},)";

            timings_file << shambase::format_printf(rstr,
                name.c_str(),
                shamsys::instance::world_rank,
                shamsys::instance::world_rank,
                now_val,
                name.c_str());
        }

        void register_end(std::string name){
            auto now_val = get_now_val<std::chrono::microseconds>();

            const char* rstr =  
R"({
    "cat": "%s",  
    "pid": %d,  
    "tid": %d, 
    "ts": %zu, 
    "ph": "E", 
    "name": "%s", 
    "args": { 
    }
},)";

            timings_file << shambase::format_printf(rstr,
                name.c_str(),
                shamsys::instance::world_rank,
                shamsys::instance::world_rank,
                now_val,
                name.c_str());
        }

        void register_end_nocoma(std::string name){
            auto now_val = get_now_val<std::chrono::microseconds>();

            const char* rstr =  
R"({
    "cat": "%s",  
    "pid": %d,  
    "tid": %d, 
    "ts": %zu, 
    "ph": "E", 
    "name": "%s", 
    "args": { 
    }
})";

            timings_file << shambase::format_printf(rstr,
                name.c_str(),
                shamsys::instance::world_rank,
                shamsys::instance::world_rank,
                now_val,
                name.c_str());
        }
    }
    

    void open_files(){
        logfiles_on = true;

        u32 world_rank = shamsys::instance::world_rank;

        timings::timings_file = std::ofstream("timings_"+std::to_string(world_rank));
        timings::timings_file << "[";
        timings::register_begin("SHAMROCK");
    }

    void close_files(){
        timings::register_end_nocoma("SHAMROCK");
        timings::timings_file << "]";
        timings::timings_file.close();
    }


    
}
