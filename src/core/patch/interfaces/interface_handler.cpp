// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "interface_handler.hpp"


template <> void LegacyInterfacehandler<f32_3, f32>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}

template <> void LegacyInterfacehandler<f64_3, f64>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}
