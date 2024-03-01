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

    struct AMRGraph {
        sycl::buffer<u32> node_link_offset;
        sycl::buffer<u32> node_links;
        u32 link_count;
    };

    struct AMRGraphLinkiterator{
        
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> node_link_offset;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> node_links;

        AMRGraphLinkiterator(AMRGraph & graph,sycl::handler & cgh):
        node_link_offset{graph.node_link_offset, cgh, sycl::read_only},
        node_links{graph.node_links, cgh, sycl::read_only} {}

        template<class Functor_iter>
        inline void for_each_object_link(const u32 & cell_id, Functor_iter &&func_it) const {
            u32 min_ids = node_link_offset[cell_id    ];
            u32 max_ids = node_link_offset[cell_id + 1];
            for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                func_it(node_links[id_s]);
            }   
        }
    };

    template<class Tvec, class TgridVec>
    struct OrientedAMRGraph {

        enum Direction {
            xp = 0,
            xm = 1,
            yp = 0,
            ym = 1,
            zp = 0,
            zm = 1,
        };

        const std::array<TgridVec, 6> offset_check{
            TgridVec{1, 0, 0},
            TgridVec{-1, 0, 0},
            TgridVec{0, 1, 0},
            TgridVec{0, -1, 0},
            TgridVec{0, 0, 1},
            TgridVec{0, 0, -1},
        };

        std::array<std::unique_ptr<AMRGraph>, 6> graph_links;
    };

    template<class Tvec, class TgridVec>
    class AMRGraphGen {

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using AMRBlock         = typename Config::AMRBlock;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        AMRGraphGen(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}


        /**
         * @brief Return the AMR graph for the block links (two blocks having a common face).
         * Could be a morton code lookup in the tree.
         * @return shambase::DistributedData<OrientedAMRGraph>
         */
        shambase::DistributedData<OrientedAMRGraph> find_AMR_block_graph_links_common_face();

        /**
         * @brief lower the AMR block graph to a cell graph
         *
         * @param oriented_block_graph
         */
        void lower_AMR_block_graph_to_cell_stencil_graph(
            shambase::DistributedData<OrientedAMRGraph> &block_graph_links);
        /**
         * @brief lower the AMR block graph to a cell graph
         *
         * @param oriented_block_graph
         */
        void lower_AMR_block_graph_to_cell_common_face_graph(
            shambase::DistributedData<OrientedAMRGraph> &block_graph_links);
    };

} // namespace shammodels::basegodunov::modules
