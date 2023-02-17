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

        void check_amr_main_fields(){

            bool correct_type = true;
            correct_type &= sched.pdl.check_field_type<Tcoord>(0);
            correct_type &= sched.pdl.check_field_type<Tcoord>(1);

            bool correct_names = true;
            correct_names &= sched.pdl.get_field<Tcoord>(0).name == "cell_min";
            correct_names &= sched.pdl.get_field<Tcoord>(1).name == "cell_max";

            if(!correct_type || !correct_names){
                throw std::runtime_error(
                    "the amr module require a layout in the form :\n"
                    "    0 : cell_min : nvar=1 type : (Coordinate type)\n"
                    "    1 : cell_max : nvar=1 type : (Coordinate type)\n\n"
                    "the current layout is : \n" +
                    sched.pdl.get_description_str()
                );
            }
        }

        explicit AMRGrid(PatchScheduler &scheduler) : sched(scheduler) {
            check_amr_main_fields();
        }

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
        scheduler::DistributedData<SplitList> gen_refinelists(
            std::function< void(u64 , patch::Patch , patch::PatchData &, sycl::buffer<u32> &) > fct
            ){

            scheduler::DistributedData<SplitList> ret;

            using namespace patch;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                sycl::queue &q = shamsys::instance::get_compute_queue();

                u32 obj_cnt = pdat.get_obj_cnt();

                sycl::buffer<u32> refine_flags(obj_cnt);

                fct(id_patch, cur_p, pdat, refine_flags);

                auto [buf, len] = shamalgs::numeric::stream_compact(q, refine_flags, obj_cnt);

                ret.add_obj(id_patch, SplitList{std::move(buf), len});
            });

            return std::move(ret);
        }



        template<class UserAcc, class Fct>
        void apply_splits(scheduler::DistributedData<SplitList> && splts , Fct && lambd){

            using namespace patch;

            

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                sycl::queue &q = shamsys::instance::get_compute_queue();

                u32 old_obj_cnt = pdat.get_obj_cnt();

                SplitList & refine_flags = splts.get(id_patch);
                pdat.expand(refine_flags.count*(split_count-1));

                
                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                    sycl::accessor index_to_ref {refine_flags.idx, cgh, sycl::read_only};

                    u32 start_index_push = old_obj_cnt;

                    constexpr u32 new_splits = split_count-1;

                    UserAcc uacc (cgh,pdat);

                    cgh.parallel_for(sycl::range<1>(refine_flags.count), [=](sycl::item<1> gid) {

                        u32 tid = gid.get_linear_id();
                        
                        u32 idx_to_refine = index_to_ref[gid];

                        std::array<u32, split_count> cells_ids;

                        cells_ids[0] = idx_to_refine;

                        #pragma unroll
                        for(u32 pid = 0; pid < new_splits; pid++){
                            cells_ids[pid+1] = start_index_push + tid*new_splits + pid;
                        }
                        
                        //lambd(idx_to_refine, cells_ids, uacc);

                    });
                });
                


            });

        }

        inline void make_base_grid(Tcoord bmin, Tcoord cell_size, std::array<u32,dim> cell_count){

            Tcoord bmax{
                bmin.x() + cell_size.x() * (cell_count[0]),
                bmin.y() + cell_size.y() * (cell_count[1]),
                bmin.z() + cell_size.z() * (cell_count[2])
            };

            sched.set_coord_domain_bound(bmin,bmax);

            if((cell_size.x() != cell_size.y()) || (cell_size.y() != cell_size.z()) ){
                logger::warn_ln("AMR Grid", "your cells aren't cube");
            }

            static_assert(dim == 3, "this is not implemented for dim != 3");

            std::array<u32, dim> patch_count;

            constexpr u32 gcd_pow2 = 1U << 31U;
            u32 gcd_cell_count;
            {
                gcd_cell_count = std::gcd(cell_count[0], cell_count[1]);
                gcd_cell_count = std::gcd(gcd_cell_count, cell_count[2]);
                gcd_cell_count = std::gcd(gcd_cell_count, gcd_pow2);
            }


            logger::debug_ln("AMRGrid","patch grid :",
                cell_count[0]/gcd_cell_count,
                cell_count[1]/gcd_cell_count,
                cell_count[2]/gcd_cell_count
            );


            sched.make_patch_base_grid<3>({
                {
                    cell_count[0]/gcd_cell_count,
                    cell_count[1]/gcd_cell_count,
                    cell_count[2]/gcd_cell_count
                }
            });

            u32 cell_tot_count = cell_count[0]*cell_count[1]*cell_count[2];


            
            sycl::buffer<Tcoord> cell_coord_min (cell_tot_count);
            sycl::buffer<Tcoord> cell_coord_max (cell_tot_count);

            shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
                
                sycl::accessor acc_min {cell_coord_min, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor acc_max {cell_coord_max, cgh, sycl::write_only, sycl::no_init};

                sycl::range<3> rnge {cell_count[0],cell_count[1],cell_count[2]};

                Tcoord sz = cell_size;

                cgh.parallel_for(rnge,[=](sycl::item<3> gid){
                    acc_min[gid.get_linear_id()] = sz* Tcoord{
                        gid.get_id(0),
                        gid.get_id(1),
                        gid.get_id(2)};
                    acc_max[gid.get_linear_id()] = sz* Tcoord{
                        gid.get_id(0)+1,
                        gid.get_id(1)+1,
                        gid.get_id(2)+1};
                });

            });



            patch::PatchData pdat (sched.pdl);
            pdat.resize(cell_tot_count);
            pdat.get_field<Tcoord>(0).override(cell_coord_min,cell_tot_count);
            pdat.get_field<Tcoord>(1).override(cell_coord_max,cell_tot_count);

            sched.allpush_data(pdat);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    //template<class Tcoord, u32 dim>
    //inline auto
    //AMRGrid<Tcoord, dim>::gen_splitlists(std::function<sycl::buffer<u32>(u64 , patch::Patch , patch::PatchData &)> fct) -> scheduler::DistributedData<SplitList> {
//
    //    scheduler::DistributedData<SplitList> ret;
//
    //    using namespace patch;
//
    //    sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
    //        sycl::queue &q = shamsys::instance::get_compute_queue();
//
    //        u32 obj_cnt = pdat.get_obj_cnt();
//
    //        sycl::buffer<u32> split_flags = fct(id_patch, cur_p, pdat);
//
    //        auto [buf, len] = shamalgs::numeric::stream_compact(q, split_flags, obj_cnt);
//
    //        ret.add_obj(id_patch, SplitList{std::move(buf), len});
    //    });
//
    //    return std::move(ret);
    //}

} // namespace shamrock::amr