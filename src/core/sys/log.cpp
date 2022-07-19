#include "log.hpp"


namespace logfiles {


    bool logfiles_on = false;


    namespace timings {
        std::ofstream timings_file;

        void register_begin(std::string name){
            auto now = std::chrono::high_resolution_clock::now();
            time_t now_val = std::chrono::high_resolution_clock::to_time_t(now);

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

            timings_file << format(rstr,
                name.c_str(),
                mpi_handler::world_rank,
                mpi_handler::world_rank,
                now_val,
                name.c_str());
        }

        void register_end(std::string name){
            auto now = std::chrono::high_resolution_clock::now();
            time_t now_val = std::chrono::high_resolution_clock::to_time_t(now);

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

            timings_file << format(rstr,
                name.c_str(),
                mpi_handler::world_rank,
                mpi_handler::world_rank,
                now_val,
                name.c_str());
        }

        void register_end_nocoma(std::string name){
            auto now = std::chrono::high_resolution_clock::now();
            time_t now_val = std::chrono::high_resolution_clock::to_time_t(now);

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

            timings_file << format(rstr,
                name.c_str(),
                mpi_handler::world_rank,
                mpi_handler::world_rank,
                now_val,
                name.c_str());
        }
    }
    

    void open_files(){
        logfiles_on = true;

        u32 world_rank = mpi_handler::world_rank;

        timings::timings_file = std::ofstream("timings_"+std::to_string(world_rank));
        timings::timings_file << "[";
        timings::register_begin("SHAMROCK");
    }

    void close_files(){timings::timings_file << "]";
        timings::register_end_nocoma("SHAMROCK");
        timings::timings_file.close();
    }


    
}
