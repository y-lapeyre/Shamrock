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
#include <stack>

namespace shambase::details {

    void add_prof_entry(std::string n, bool is_start);
    void dump_profiling(u32 world_rank);

    inline std::stack<SourceLocation> call_stack;

    struct BasicStackEntry {
        SourceLocation loc;
        bool do_timer;

        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer) {
            if (do_timer)
                add_prof_entry(loc.functionName, true);
            call_stack.emplace(loc);
        }

        inline ~BasicStackEntry() {
            if (do_timer)
                add_prof_entry(call_stack.top().functionName, false);
            call_stack.pop();
        }
    };

    struct NamedBasicStackEntry {
        SourceLocation loc;
        bool do_timer;
        std::string name;

        inline NamedBasicStackEntry(std::string name,
                                    bool do_timer        = true,
                                    SourceLocation &&loc = SourceLocation{})
            : name(name), loc(loc), do_timer(do_timer) {
            if (do_timer)
                add_prof_entry(name, true);
            call_stack.emplace(loc);
        }

        inline ~NamedBasicStackEntry() {
            if (do_timer)
                add_prof_entry(name, false);
            call_stack.pop();
        }
    };

} // namespace shambase::details

namespace shambase {

    /**
     * @brief get the formatted callstack
     *
     * @return std::string
     */
    std::string fmt_callstack();

} // namespace shambase

using StackEntry      = shambase::details::BasicStackEntry;
using NamedStackEntry = shambase::details::NamedBasicStackEntry;