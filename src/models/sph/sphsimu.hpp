#pragma once


#include "aliases.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"


class SimulationSPH{

    f64 current_time;

    

    template<class Stepper> void evolve(PatchScheduler &sched, Stepper & stepper);

};