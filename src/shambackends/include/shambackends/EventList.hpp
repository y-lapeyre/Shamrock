// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EventList.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/DeviceContext.hpp"

namespace sham {

    class EventList {
        public:
        void apply_dependancy(sycl::handler &h) { h.depends_on(events); }

        void wait() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait();
            }
            consumed = true;
        }

        void wait_and_throw() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait_and_throw();
            }
            consumed = true;
        }

        void add_event(sycl::event e) { events.push_back(e); }

        EventList(SourceLocation loc = SourceLocation{}) : loc_build(loc) {}

        ~EventList();

        private:
        std::vector<sycl::event> events = {};
        bool consumed                   = false;
        SourceLocation loc_build;

        friend class DeviceQueue;
    };
} // namespace sham
