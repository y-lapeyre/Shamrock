#pragma once

#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "models/sph/integrators/leapfrog.hpp"
#include <string>

namespace models::sph {

    template<class flt, class u_morton, class Kernel>
    class BasicSPHGas {

        using vec3 = sycl::vec<flt, 3>;
        using Stepper = integrators::sph::LeapfrogGeneral<flt, Kernel, u_morton>;

        static constexpr flt htol_up_tol  = 1.4;
        static constexpr flt htol_up_iter = 1.2;

        flt cfl_cour  = -1;
        flt cfl_force = -1;
        flt gpart_mass = -1;

        //TODO change SimBoxInfo into boundary condition to extract it from the models, then we can throw errors if the BC is not implemented for the model
        static constexpr bool periodic_bc = true;

        static constexpr flt eos_cs = 1;

        void check_valid();

        public:

        void init();
        void evolve(PatchScheduler &sched, f64 &step_time);
        void dump(std::string prefix);
        void restart_dump(std::string prefix);
        void close();


        inline void set_cfl_cour(flt Ccour){cfl_cour = Ccour;}
        inline void set_cfl_force(flt Cforce){cfl_force = Cforce;}
        inline void set_particle_mass(flt pmass){gpart_mass = pmass;}

    };


} // namespace models::sph