// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

//%Impl status : Deprecated



#include "aliases.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"


class SimulationSPH{

    f64 current_time;

    

    template<class Stepper> void evolve(PatchScheduler &sched, Stepper & stepper);

};