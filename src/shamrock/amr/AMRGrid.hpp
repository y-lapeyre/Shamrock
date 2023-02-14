// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "AMRCell.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shamrock/scheduler/DistributedData.hpp"





namespace shamrock::amr {

    struct SplitList{
        sycl::buffer<u32> idx;
        u32 count;
    };

    



    template<class Tcoord, u32 dim>
    class AMRGrid{

        using CellCoord = AMRCellCoord<Tcoord, dim>;

        static constexpr u32 dimension   = dim;
        static constexpr u32 split_count = CellCoord::splts_count;

        PatchScheduler & sched;

        explicit AMRGrid(PatchScheduler & scheduler) : sched(scheduler){}

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
        scheduler::DistributedData<SplitList> gen_splitlists(Fct && f){

            scheduler::DistributedData<SplitList> ret;

            using namespace patch;

            sched.for_each_patch_data(
                [&](u64 id_patch, Patch cur_p, PatchData &pdat) {

                    sycl::queue & q = shamsys::instance::get_compute_queue();
                    
                    u32 obj_cnt = pdat.get_obj_cnt();

                    sycl::buffer<u32> split_flags = f(id_patch, cur_p,pdat);

                    auto [buf,len] = shamalgs::numeric::stream_compact(q, split_flags, obj_cnt);

                    ret.add_obj(id_patch,SplitList {
                        std::move(buf),len  
                    });

                }
            );

        }

        

    };


}