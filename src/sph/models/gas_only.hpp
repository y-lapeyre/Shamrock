#pragma once

#include "patchscheduler/scheduler_mpi.hpp"

namespace models::sph {

template<class flt, class u_morton, class Kernel>
class GasOnly{
    flt part_mass;

    void step(PatchScheduler &sched, f64 &step_time) ;

};

}