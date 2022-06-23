#include "interface_handler.hpp"


template <> void InterfaceHandler<f32_3, f32>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}

template <> void InterfaceHandler<f64_3, f64>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}
