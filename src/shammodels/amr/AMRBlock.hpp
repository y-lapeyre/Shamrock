// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/integer.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/type_aliases.hpp"

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

        static constexpr u32 block_size = shambase::pow_constexpr<dim>(Nside);

        inline constexpr u32 get_index(std::array<u32, dim> coord) noexcept {
            static_assert(dim < 4, "not implemented above dim 3");

            if constexpr (dim == 1){
                return coord[0];
            }

            if constexpr (dim == 2){
                return coord[0] + Nside * coord[1];
            }

            if constexpr (dim == 3){
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
            }

            return {};
        }

        
    };

} // namespace shammodels::amr