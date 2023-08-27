// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRBlock.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief utility to manipulate AMR blocks
 */

#include "shambase/integer.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/type_aliases.hpp"
#include <array>

namespace shammodels::amr {

    /**
     * @brief utility class to handle AMR blocks
     *
     * @tparam Tvec
     * @tparam Nside
     */
    template<class Tvec, class TgridVec, u32 Nside>
    struct AMRBlock {

        static constexpr u32 dim = shambase::VectorProperties<TgridVec>::dimension;

        static constexpr u32 side_size = Nside;

        static constexpr u32 block_size = shambase::pow_constexpr<dim>(Nside);

        /**
         * @brief Get the local index within the AMR block
         *
         * @param coord wanted integer coordinates
         * @return constexpr u32 the index
         */
        inline static constexpr u32 get_index(std::array<u32, dim> coord) noexcept {
            static_assert(dim < 5, "not implemented above dim 4");

            if constexpr (dim == 1) {
                return coord[0];
            }

            if constexpr (dim == 2) {
                return coord[0] + Nside * coord[1];
            }

            if constexpr (dim == 3) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
            }

            if constexpr (dim == 4) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2] +
                       Nside * Nside * Nside * coord[3];
            }

            return {};
        }

        template<class Func>
        inline static void
        for_each_cell_in_block(Tvec delta_cell, Func && functor) noexcept {
            static_assert(dim == 3, "implemented only in dim 3");
            for (u32 ix = 0; ix < side_size; ix++) {
                for (u32 iy = 0; iy < side_size; iy++) {
                    for (u32 iz = 0; iz < side_size; iz++) {
                        u32 i          = get_index({ix, iy, iz});
                        Tvec delta_val = delta_cell * Tvec{ix, iy, iz};
                        functor(i, delta_val);
                    }
                }
            }
        }

        inline static void parralel_for_block(sycl::buffer<TgridVec> &buf_cell_min,
                                              sycl::buffer<TgridVec> &buf_cell_max,
                                              sycl::handler &cgh,
                                              std::string name,
                                              u32 block_cnt) {
            // we use one thread per subcell because :
            // double load are avoided because of contiguous L2 cache hit
            // and CF perf opti for GPU, finer threading lead to better latency hidding
            shambase::parralel_for(cgh, block_cnt * block_size, name, [=](u64 id_g) {

            });
        }
    };

} // namespace shammodels::amr