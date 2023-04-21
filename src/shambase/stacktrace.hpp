// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "SourceLocation.hpp"
#include "shambase/time.hpp"
#include <sstream>
#include <stack>


namespace shambase::details {

    inline std::stack<SourceLocation> call_stack;

    struct BasicStackEntry {
        SourceLocation loc;

        inline BasicStackEntry(SourceLocation &&loc = SourceLocation{}) : loc(loc) {
            shambase::details::call_stack.emplace(loc);
        }

        inline ~BasicStackEntry() { shambase::details::call_stack.pop(); }
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