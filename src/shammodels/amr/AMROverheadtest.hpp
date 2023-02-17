// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamrock/amr/AMRGrid.hpp"






class AMRTestModel {
    public:
    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;
    Grid &grid;

    explicit AMRTestModel(Grid &grd) : grid(grd) {}

    inline void step() {

        using namespace shamrock::patch;

        auto splits = grid.gen_refinelists(
            [](u64 id_patch, Patch p, PatchData &pdat, sycl::buffer<u32> &refine_flags) {
                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor refine_acc{refine_flags, cgh, sycl::write_only, sycl::no_init};

                    sycl::accessor cell_low_bound{
                        *pdat.get_field<u64_3>(0).get_buf(), cgh, sycl::read_only};
                    sycl::accessor cell_high_bound{
                        *pdat.get_field<u64_3>(1).get_buf(), cgh, sycl::read_only};

                    cgh.parallel_for(sycl::range<1>(pdat.get_obj_cnt()), [=](sycl::item<1> gid) {
                        u64_3 low_bound  = cell_low_bound[gid];
                        u64_3 high_bound = cell_high_bound[gid];

                        using namespace shammath;

                        bool should_refine =
                            is_in_half_open(low_bound, u64_3{1, 1, 1}, u64_3{10, 10, 10}) &&
                            is_in_half_open(high_bound, u64_3{1, 1, 1}, u64_3{10, 10, 10});

                        refine_acc[gid] = should_refine;

                    });
                });
            }
        );





        
        class RefineCellAccessor{public:

            sycl::accessor<f32, 1, sycl::access::mode::read_write, sycl::target::device> field;

            RefineCellAccessor(sycl::handler &cgh, PatchData & pdat) :
                field{*pdat.get_field<f32>(2).get_buf(), cgh, sycl::read_write} 
                {}

        };


        grid.apply_splits<RefineCellAccessor>(std::move(splits),
            [](u32 cur_idx, std::array<u32,8> new_cells, RefineCellAccessor acc){

                f32 val = acc.field[cur_idx];

                #pragma unroll
                for(u32 pid = 0; pid < 8; pid++){
                    acc.field[new_cells[pid]] = val;
                }
                
            }
        );


    }
};