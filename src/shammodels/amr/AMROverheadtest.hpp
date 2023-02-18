// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/memory/memory.hpp"
#include "shammath/sycl_utilities.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamsys/legacy/log.hpp"
#include <vector>

class AMRTestModel {
    public:
    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;
    Grid &grid;

    explicit AMRTestModel(Grid &grd) : grid(grd) {}

    class RefineCritCellAccessor {
        public:
        sycl::accessor<u64_3, 1, sycl::access::mode::read, sycl::target::device> cell_low_bound;
        sycl::accessor<u64_3, 1, sycl::access::mode::read, sycl::target::device> cell_high_bound;

        RefineCritCellAccessor(
            sycl::handler &cgh,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchData &pdat
        )
            : cell_low_bound{*pdat.get_field<u64_3>(0).get_buf(), cgh, sycl::read_only},
              cell_high_bound{*pdat.get_field<u64_3>(1).get_buf(), cgh, sycl::read_only} {}
    };

    class RefineCellAccessor {
        public:
        sycl::accessor<u32, 1, sycl::access::mode::read_write, sycl::target::device> field;

        RefineCellAccessor(sycl::handler &cgh, shamrock::patch::PatchData &pdat)
            : field{*pdat.get_field<u32>(2).get_buf(), cgh, sycl::read_write} 
            {}
    };

    inline void dump_patch(u64 id){
        
        using namespace shamrock::patch;
        using namespace shamalgs::memory;

        PatchData & pdat = grid.sched.patch_data.owned_data.at(id);

        std::vector<u64_3> mins = buf_to_vec(*pdat.get_field<u64_3>(0).get_buf(), pdat.get_obj_cnt());
        std::vector<u64_3> maxs = buf_to_vec(*pdat.get_field<u64_3>(1).get_buf(), pdat.get_obj_cnt());


        logger::raw_ln("----- dump");
        for(u32 i = 0; i < mins.size(); i++){
            logger::raw_ln(mins[i],maxs[i]);
        }
        logger::raw_ln("-----");
    } 


    /**
     * @brief does the refinment step of the AMR
     * 
     */
    inline void refine() {

        dump_patch(4);
        auto splits = grid.gen_refine_list<RefineCritCellAccessor>(
            [](u32 cell_id, RefineCritCellAccessor acc) -> u32 {
                u64_3 low_bound  = acc.cell_low_bound[cell_id];
                u64_3 high_bound = acc.cell_high_bound[cell_id];

                using namespace shammath;

                bool should_refine = is_in_half_open(low_bound, u64_3{2, 2, 2}, u64_3{8, 8, 8}) &&
                                     is_in_half_open(high_bound, u64_3{2, 2, 2}, u64_3{8, 8, 8});

                should_refine = should_refine && (high_bound.x() - low_bound.x() > 1);
                should_refine = should_refine && (high_bound.y() - low_bound.y() > 1);
                should_refine = should_refine && (high_bound.z() - low_bound.z() > 1);

                return should_refine;
            }
        );

        

        grid.apply_splits<RefineCellAccessor>(
            std::move(splits),

            [](u32 cur_idx,
               Grid::CellCoord cur_coords,
               std::array<u32, 8> new_cells,
               std::array<Grid::CellCoord, 8> new_cells_coords,
               RefineCellAccessor acc) {
                
                u32 val = acc.field[cur_idx];

#pragma unroll
                for (u32 pid = 0; pid < 8; pid++) {
                    acc.field[new_cells[pid]] = val;
                }
            }

        );




        dump_patch(4);
        

        






    }

    inline void derefine(){
        auto merge = grid.gen_merge_list<RefineCritCellAccessor>(
            [](u32 cell_id, RefineCritCellAccessor acc) -> u32 {
                u64_3 low_bound  = acc.cell_low_bound[cell_id];
                u64_3 high_bound = acc.cell_high_bound[cell_id];

                using namespace shammath;

                bool should_merge = is_in_half_open(low_bound, u64_3{2, 2, 2}, u64_3{8, 8, 8}) &&
                                     is_in_half_open(high_bound, u64_3{2, 2, 2}, u64_3{8, 8, 8});

                return should_merge;
            }
        );

        grid.apply_merge<RefineCellAccessor>(
            std::move(merge),

            [](
                 std::array<u32, 8> old_cells,
                 std::array<Grid::CellCoord, 8> old_coords,
                 u32 new_cell,
                 Grid::CellCoord new_coord,
                
               RefineCellAccessor acc) {
                
                u32 accum = 0;

                #pragma unroll
                for (u32 pid = 0; pid < 8; pid++) {
                    accum += acc.field[old_cells[pid]];
                }

                acc.field[new_cell] = accum / 8;

            }

        );
        dump_patch(4);
    }

    inline void step() {
        using namespace shamrock::patch;
        refine();
        derefine();
    }
};