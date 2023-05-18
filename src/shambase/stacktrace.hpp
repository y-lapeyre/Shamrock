// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "SourceLocation.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include <sstream>
#include <stack>
#include <vector>


namespace shambase::details {

    inline Timer global_timer = Timer{};

    struct ChomeProfileEntry{
        std::string name;
        u64 time_val;
        bool is_start;

        std::string format_start(u32 world_rank){
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "B", 
                "name": "%s", 
                "args": { 
                }
            })", name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());
        }

        std::string format_end(u32 world_rank){
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "E", 
                "name": "%s", 
                "args": { 
                }
            })", name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());
        }

        std::string format(u32 world_rank){
            if(is_start){
                return format_start(world_rank);
            }else{
                return format_end(world_rank);
            }
        }
    };

    inline std::vector<ChomeProfileEntry> chome_prof;

    inline void add_prof_entry(std::string n,bool is_start){
        global_timer.end();
        chome_prof.push_back(
            ChomeProfileEntry{n,static_cast<u64>(global_timer.elasped_sec()*1e9),is_start}
        );
    }

    inline void dump_profiling(u32 world_rank){
        std::ofstream outfile ("timings_"+std::to_string(world_rank));
        outfile <<"[";

        u32 len = chome_prof.size();

        for (u32 i = 0; i < len; i++) {
            outfile << chome_prof[i].format(world_rank);
            if(i != len-1){
                outfile << ",";
            } 
        }

        outfile <<"]";
        outfile.close();
    }








    inline std::stack<SourceLocation> call_stack;

    struct BasicStackEntry {
        SourceLocation loc;
        bool do_timer;

        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{}) : loc(loc), do_timer(do_timer) {
            add_prof_entry(loc.functionName,true);
            call_stack.emplace(loc);
        }

        inline ~BasicStackEntry() { 
            add_prof_entry(call_stack.top().functionName,false);
            call_stack.pop(); 
        }
    };

    

} // namespace shambase::details

namespace shambase {

    inline std::string fmt_callstack(){
        std::stack<SourceLocation> cpy = details::call_stack;

        std::vector<std::string> lines;

        while (!cpy.empty( ) )
        {
            SourceLocation l = cpy.top( );
            lines.push_back(l.format_one_line_func());
            cpy.pop( );
        }

        std::reverse(lines.begin(), lines.end());
        
        std::stringstream ss;
        for (u32 i = 0; i < lines.size(); i++) {
            ss <<shambase::format(" {:2} : {}\n", i,lines[i]);
        }
        

        return ss.str();

    }

}

using StackEntry = shambase::details::BasicStackEntry;