// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include "aliases.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamrock/legacy/utils/string_utils.hpp"
#include "shamrock/legacy/utils/time_utils.hpp"

namespace logfiles {
    inline bool dump_timings = true;
    inline std::vector<MPI_File> timing_files;

    inline void open_log_files(){

        using namespace shamsys::instance;

        if(dump_timings){
            timing_files.resize(world_size);
            for(u32 id = 0 ; id < world_size ; id++){
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
} // namespace logfiles



namespace timings {

    enum timingtype{
        function,
        sycl,
        mpi
    };

    struct LogTimers {
        std::string name;
        f64 time;
        timingtype timekind;
        u32 active_timers;

        bool is_bandwidth;
        f64 data_transfered;
        f64 bandwith;
    };

    inline std::vector<LogTimers> timer_log;
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
            logfiles::timings::register_begin(n);
        }

        inline void stop(){
            //shamsys::instance::get_compute_queue().wait();
            time.end();
            active_timers --;
            timer_log.push_back(LogTimers{
                name,
                time.nanosec/1.e9, 
                kind,
                active_timers,
                false,
                -1,
                -1});
            logfiles::timings::register_end(name);
        }

        inline void stop(u64 data_transfered){
            //shamsys::instance::get_compute_queue().wait();
            time.end();
            active_timers --;
            timer_log.push_back(LogTimers{
                name,
                time.nanosec/1.e9, 
                kind,
                active_timers,
                true,
                f64(data_transfered),
                (u64(data_transfered))/(time.nanosec/1.e9)});
            logfiles::timings::register_end(name);
        }
    };

    inline NamedTimer start_timer(std::string name,timingtype t){
        return NamedTimer(name,t);
    }

    inline void dump_timings(std::string header){

        header = "\n\n" + header +"\n\n";

        if(logfiles::dump_timings){
            MPI_Status st;
            mpi::file_write(logfiles::timing_files[shamsys::instance::world_rank], header.data(), header.size(), mpi_type_u8, &st);

            f64 total = 0;
            for(auto & a : timer_log){
                if(a.active_timers == 0) total += a.time;
            }

            for(auto & a : timer_log){

                std::string str = "";
                for(u32 i = 0; i < a.active_timers; i ++){
                    str += "  ";
                }str += a.name;

                
                
                std::string out = format("%-50s %2.9f %3.1f ",str.c_str(), a.time,100*a.time/total);

                if (a.is_bandwidth) {
                    out += readable_sizeof(a.data_transfered);
                    out += " Bandwith : " + readable_sizeof(a.bandwith) + ".s-1\n";
                }else {
                    out += "\n";
                }
                


                mpi::file_write(logfiles::timing_files[shamsys::instance::world_rank], out.data(), out.size(), mpi_type_u8, &st);
            }
        }
        timer_log.clear();
    }


};
