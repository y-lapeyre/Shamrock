// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file StencilGenerator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
#include "shammodels/amr/AMRBlockStencil.hpp"
#include "shammodels/amr/AMRCellStencil.hpp"


namespace shammodels::basegodunov::modules {


    template<class Tvec, class TgridVec>
    class StencilGenerator {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec,TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;
        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        StencilGenerator(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        enum StencilOffsets{
            xp1 = 0,
            xm1 = 1,
            yp1 = 2,
            ym1 = 3,
            zp1 = 4,
            zm1 = 5,
        };
        static constexpr u32 stencil_offset_count = 6;

        private:

        using block_stencil_el_buf = sycl::buffer<amr::block::StencilElement>;
        using cell_stencil_el_buf = sycl::buffer<amr::cell::StencilElement>;

        using dd_block_stencil_el_buf = shambase::DistributedData<block_stencil_el_buf>;
        using dd_cell_stencil_el_buf = shambase::DistributedData<cell_stencil_el_buf>;

        dd_block_stencil_el_buf compute_block_stencil_slot(i64_3 relative_pos, StencilOffsets result_offset);
        cell_stencil_el_buf lower_block_slot_to_cell(i64_3 relative_pos, StencilOffsets result_offset, block_stencil_el_buf & block_stencil_el);
        dd_cell_stencil_el_buf lower_block_slot_to_cell(i64_3 relative_pos, StencilOffsets result_offset, dd_block_stencil_el_buf & block_stencil_el);
        
        
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::basegodunov::modules