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
    template<class Tvec, u32 Nside>
    struct AMRBlock {

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

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


    };

} // namespace shammodels::amr