#pragma once

#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "models/sph/integrators/leapfrog.hpp"
#include <string>

namespace models::sph {

    template<class flt, class u_morton, class Kernel>
    class BasicSPHGas {

        using vec3 = sycl::vec<flt, 3>;
        using Stepper = integrators::sph::LeapfrogGeneral<flt, Kernel, u_morton>;

        flt cfl_cour  = 0.1;
        flt cfl_force = 0.1;

        flt htol_up_tol  = 1.4;
        flt htol_up_iter = 1.2;

        public:

        void init();
        void evolve(PatchScheduler &sched, f64 &step_time);
        void dump(std::string prefix);
        void restart_dump(std::string prefix);
        void close();


        inline void set_cfl_cour(flt Ccour){cfl_cour(Ccour);}
        inline void set_cfl_force(flt Cforce){cfl_force(Cforce);}

    };

} // namespace models::sph