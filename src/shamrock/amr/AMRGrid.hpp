// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "AMRCell.hpp"
#include "aliases.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shamrock/math/integerManip.hpp"
#include "shamrock/scheduler/DistributedData.hpp"

namespace shamrock::amr {

    struct SplitList {
        sycl::buffer<u32> idx;
        u32 count;
    };

    /**
     * @brief The AMR grid only sees the grid as an integer map
     * 
     * @tparam Tcoord 
     * @tparam dim 
     */
    template<class Tcoord, u32 dim>
    class AMRGrid {public:

        PatchScheduler & sched;

        using CellCoord = AMRCellCoord<Tcoord, dim>;
        static constexpr u32 dimension   = dim;
        static constexpr u32 split_count = CellCoord::splts_count;

        explicit AMRGrid(PatchScheduler &scheduler) : sched(scheduler) {}

        /**
         * @brief generate split lists for all patchdata owned by the node
         * ~~~~~{.cpp}
         *
         * auto split_lists = grid.gen_splitlists(
         *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) -> sycl::buffer<u32> {
         *          generate the buffer saying which cells should split
         *     }
         * );
         *
         * ~~~~~
         *
         * @tparam Fct
         * @param f
         * @return scheduler::DistributedData<SplitList>
         */
        template<class Fct>
        scheduler::DistributedData<SplitList> gen_splitlists(Fct &&f);

        inline void make_base_grid(Tcoord bmin, Tcoord bmax, std::array<u32,dim> cell_count){

            sched.set_coord_domain_bound(bmin,bmax);

            static_assert(dim == 3, "this is not implemented for dim != 3");

            std::array<u32, dim> patch_count;


            u32 gcd_cell_count;
            {
                gcd_cell_count = std::gcd(cell_count[0], cell_count[1]);
                gcd_cell_count = std::gcd(gcd_cell_count, cell_count[2]);
            }


            sched.make_patch_base_grid({
                {
                    cell_count[0]/gcd_cell_count,
                    cell_count[1]/gcd_cell_count,
                    cell_count[2]/gcd_cell_count
                }
            });

            //check cells are squared
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Tcoord, u32 dim>
    template<class Fct>
    inline auto
    AMRGrid<Tcoord, dim>::gen_splitlists(Fct &&f) -> scheduler::DistributedData<SplitList> {

        scheduler::DistributedData<SplitList> ret;

        using namespace patch;

        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            sycl::queue &q = shamsys::instance::get_compute_queue();

            u32 obj_cnt = pdat.get_obj_cnt();

            sycl::buffer<u32> split_flags = f(id_patch, cur_p, pdat);

            auto [buf, len] = shamalgs::numeric::stream_compact(q, split_flags, obj_cnt);

            ret.add_obj(id_patch, SplitList{std::move(buf), len});
        });

        return std::move(ret);
    }

} // namespace shamrock::amr