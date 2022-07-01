#pragma once

#include "aliases.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"

#include "core/patch/base/patch.hpp"
#include "core/patch/utility/serialpatchtree.hpp"

#include "models/generic/setup/generators.hpp"


namespace models::sph {

    template<class flt, class Kernel>
    class SetupSPH {

        using vec = sycl::vec<flt,3>;

        bool periodic_mode;
        u64 part_cnt = 0;

        public:

        void init(PatchScheduler & sched);

        void set_boundaries(bool periodic){
            periodic_mode = periodic;
        }

        inline std::tuple<vec,vec> get_ideal_box(flt dr, std::tuple<vec,vec> box){
            return get_ideal_fcc_box(dr, box);
        }

        inline vec get_box_dim(flt dr, u32 xcnt, u32 ycnt, u32 zcnt){
            return get_box_dim(dr, xcnt, ycnt, zcnt);
        }
        

        void add_particules_fcc(PatchScheduler & sched, flt dr, std::tuple<vec,vec> box);

        inline flt set_total_mass(flt tot_mass){
            u64 part = 0;
            mpi::allreduce(&part_cnt, &part, 1, mpi_type_u64, MPI_SUM, MPI_COMM_WORLD);
            return tot_mass/part;
        }

    };


} // namespace models::sph