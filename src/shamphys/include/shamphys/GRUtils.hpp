// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GRUtils.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambackends/math.hpp"
#include "shamunits/Constants.hpp"
#include <experimental/mdspan>
#include <shambackends/sycl.hpp>

namespace shamphys {

    template<
        class Tvec,
        class Tscal,
        class SizeType,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    struct GR_physics {

        inline static constexpr Tscal get_alpha(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            // alpha = 1 / sqrt(-g00)
            return 1. / sycl::sqrt(-gcov(0, 0));
        }

        inline static constexpr Tvec get_betaUP(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            Tscal alpha  = get_alpha(gcov);
            Tscal alpha2 = alpha * alpha;
            Tvec betaUP  = {0., 0., 0.};

            betaUP[0] = gcov(0, 1) * alpha2;
            betaUP[1] = gcov(0, 2) * alpha2;
            betaUP[2] = gcov(0, 3) * alpha2;

            return betaUP;
        }

        inline static constexpr Tscal GR_dot(
            Tvec a,
            Tvec b,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
            // TODO
        }

        inline static constexpr Tscal get_gamma(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij;
            for (int i = 1; i < 4; i++) {
                for (int j = 1; j < 4; j++) {
                    gamma_ij(i, j) = gcov(i, j);
                }
            }

            return mat_det_33(gamma_ij);
        }

        inline static constexpr Tscal get_U0(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            // U0 = gamma / alpha
            Tscal alpha = get_alpha(gcov);
            Tscal gamma = get_gamma(gcov);
            return gamma / alpha;
        }

        inline static constexpr Tvec get_V(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            Tscal alpha = get_alpha(gcov);
            Tvec betaUP = get_betaUP(gcov);
            Tvec V      = (vxyz + betaUP) / alpha;

            return V;
        }

        /// placeholder function for more complex metrics
        inline static constexpr Tscal get_sqrt_g(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            // for unusual metric, need to compute the determinant
            // for now (Kerr), this is enough
            return 1;
        }

        /// placeholder function for more complex metrics
        inline static constexpr Tscal get_sqrt_gamma(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            // for unusual metric, need to compute the determinant of the spatial metric
            // for now (Kerr), this is enough
            return 1;
        }
    };

} // namespace shamphys