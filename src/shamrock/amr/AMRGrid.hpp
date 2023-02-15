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

            u32 max_lin_cell_count = 0;
            for(u32 i = 0 ; i < dim; i++){
                max_lin_cell_count = sycl::max(max_lin_cell_count, cell_count[i]);
            }

            u64 coord_div_fact = math::int_manip::get_next_pow2_val(max_lin_cell_count);

            u64 sz_root_patch = PatchScheduler::max_axis_patch_coord_lenght/coord_div_fact;

            
            std::vector<patch::PatchCoord> coords;
            for(u32 x = 0; x < cell_count[0]; x++){
                for(u32 y = 0; y < cell_count[1]; y++){
                    for(u32 z = 0; z < cell_count[2]; z++){
                        patch::PatchCoord coord;

                        coord.x_min = sz_root_patch*(x);
                        coord.y_min = sz_root_patch*(y);
                        coord.z_min = sz_root_patch*(z);
                        coord.x_max = sz_root_patch*(x+1)-1;
                        coord.y_max = sz_root_patch*(y+1)-1;
                        coord.z_max = sz_root_patch*(z+1)-1;

                        coords.push_back(coord);
                    }
                }
            }

            sched.add_root_patches(coords);

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