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

#include "shambackends/sycl.hpp"

namespace shammodels::basegodunov::modules {
    
    struct NeighGraph {
        sycl::buffer<u32> node_link_offset;
        sycl::buffer<u32> node_links;
        u32 link_count;
    };

    struct NeighGraphLinkiterator {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> node_link_offset;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> node_links;

        NeighGraphLinkiterator(NeighGraph &graph, sycl::handler &cgh)
            : node_link_offset{graph.node_link_offset, cgh, sycl::read_only},
              node_links{graph.node_links, cgh, sycl::read_only} {}

        template<class Functor_iter>
        inline void for_each_object_link(const u32 &cell_id, Functor_iter &&func_it) const {
            u32 min_ids = node_link_offset[cell_id];
            u32 max_ids = node_link_offset[cell_id + 1];
            for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                func_it(node_links[id_s]);
            }
        }

        template<class Functor_iter>
        inline u32 for_each_object_link_cnt(const u32 &cell_id, Functor_iter &&func_it) const {
            u32 min_ids = node_link_offset[cell_id];
            u32 max_ids = node_link_offset[cell_id + 1];
            for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                func_it(node_links[id_s]);
            }
            return max_ids - min_ids;
        }
    };

    using AMRGraph = NeighGraph;
    using AMRGraphLinkiterator = NeighGraphLinkiterator;

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
} // namespace shammodels::basegodunov::modules