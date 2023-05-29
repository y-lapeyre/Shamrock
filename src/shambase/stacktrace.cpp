// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "stacktrace.hpp"
#include "shambase/time.hpp"
#include <sstream>

namespace shambase::details {

    Timer global_timer = Timer{};

    struct ChromeProfileEntry {
        std::string name;
        u64 time_val;
        bool is_start;

        std::string format(u32 world_rank);
    };

    std::vector<ChromeProfileEntry> chome_prof;

    void add_prof_entry(std::string n, bool is_start) {
        global_timer.end();
        chome_prof.push_back(
            ChromeProfileEntry{n, static_cast<u64>(global_timer.elasped_sec() * 1e6), is_start});
    }

    std::string ChromeProfileEntry::format(u32 world_rank) {
        if (is_start) {
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "B", 
                "name": "%s", 
                "args": { 
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());

        } else {
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "E", 
                "name": "%s", 
                "args": { 
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());
        }
    }

    void dump_profiling(u32 world_rank) {
        std::ofstream outfile("timings_" + std::to_string(world_rank));
        outfile << "[";

        u32 len = chome_prof.size();

        for (u32 i = 0; i < len; i++) {
            outfile << chome_prof[i].format(world_rank);
            if (i != len - 1) {
                outfile << ",";
            }
        }

        outfile << "]";
        outfile.close();
    }

} // namespace shambase::details