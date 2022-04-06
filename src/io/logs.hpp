#pragma once


#include <string>
#include <tuple>
#include <vector>
#include "aliases.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "utils/string_utils.hpp"
#include "utils/time_utils.hpp"

namespace logfiles {
    inline bool dump_timings = true;
    inline std::vector<MPI_File> timing_files;

    inline void open_log_files(){
        if(dump_timings){
            timing_files.resize(mpi_handler::world_size);
            for(u32 id = 0 ; id < mpi_handler::world_size ; id++){
                std::string fname = "timing_"+std::to_string(id) + ".txt";

                int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &timing_files[id]);

                if (rc) {
                    printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
                }
            }
        }
    }

    inline void close_log_files(){
        for(auto & a : timing_files){
            mpi::file_close(&a);
        }
    }
}



namespace timings {

    enum timingtype{
        function,
        sycl,
        mpi
    };

    inline std::vector<std::tuple<std::string,f64,timingtype,u32>> timer_log;
    inline u32 active_timers = 0;

    class NamedTimer{
    
        private:
        

        Timer time;
        std::string name;
        timingtype kind;

        public:
        NamedTimer(std::string n, timingtype t){
            name = n;
            kind = t;
            active_timers ++;
            time.start();
        }

        inline void stop(){
            time.end();
            active_timers --;
            timer_log.push_back({name,time.nanosec/1.e9, kind,active_timers});
        }
    };

    inline NamedTimer start_timer(std::string name,timingtype t){
        return NamedTimer(name,t);
    }

    inline void dump_timings(std::string header){

        header = "\n\n" + header +"\n\n";

        if(logfiles::dump_timings){
            MPI_Status st;
            mpi::file_write(logfiles::timing_files[mpi_handler::world_rank], header.data(), header.size(), mpi_type_u8, &st);

            f64 total = 0;
            for(auto & [name,time,kind,indent] : timer_log){
                if(indent == 0) total += time;
            }

            for(auto & [name,time,kind,indent] : timer_log){

                std::string str = "";
                for(u32 i = 0; i < indent; i ++){
                    str += "  ";
                }str += name;
                
                std::string out = format("%-50s %2.9f %3.1f\n",str.c_str(), time,100*time/total);
                mpi::file_write(logfiles::timing_files[mpi_handler::world_rank], out.data(), out.size(), mpi_type_u8, &st);
            }
        }
        timer_log.clear();
    }


};
