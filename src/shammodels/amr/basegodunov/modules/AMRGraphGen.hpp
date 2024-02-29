// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRGraphGen.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/sycl_utils/vectorProperties.hpp"

#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"

namespace shammodels::basegodunov::modules {

    struct AMRBlockGraph{
        sycl::buffer<u32> node_link_offset;
        sycl::buffer<u32> node_links;
        u32 link_count;
    };

    template<class Tvec, class TgridVec>
    struct OrientedAMRBlockGraph{

        enum Direction{
            xp = 0,
            xm = 1,
            yp = 0,
            ym = 1,
            zp = 0,
            zm = 1,
        };

        const std::array<TgridVec,6 > offset_check{
            TgridVec{ 1,0,0},
            TgridVec{-1,0,0},
            TgridVec{0, 1,0},
            TgridVec{0,-1,0},
            TgridVec{0,0, 1},
            TgridVec{0,0,-1},
        };

        std::array<std::unique_ptr<AMRBlockGraph>,6> block_graph_links;
    };

    template<class Tvec, class TgridVec>
    class AMRGraphGen {

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config   = SolverConfig<Tvec, TgridVec>;
        using Storage  = SolverStorage<Tvec, TgridVec, u64>;
        using AMRBlock = typename Config::AMRBlock;
        using OrientedAMRBlockGraph = OrientedAMRBlockGraph<Tvec,TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        AMRGraphGen(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        shambase::DistributedData<OrientedAMRBlockGraph> find_AMR_block_graph_links();

        void lower_AMR_block_graph_to_cell(shambase::DistributedData<OrientedAMRBlockGraph> & oriented_block_graph);

    };

} // namespace shammodels::basegodunov::modules
