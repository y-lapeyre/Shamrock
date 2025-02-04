// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mhd.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambackends/math.hpp"

namespace shamphys {

    template<class Tvec, class Tscal>
    struct MHD_physics {
        public:
        Tscal v_alfven = [](Tvec B, Tscal rho, Tscal mu_0) {
            return sycl::sqrt(sycl::dot(B, B) / (mu_0 * rho));
        };

        Tscal v_shock_mhd = [this](Tscal cs, Tvec B, Tscal rho, Tscal mu_0) {
            return sycl::sqrt(cs * cs + v_alfven(B, rho, mu_0) * v_alfven(B, rho, mu_0));
        };

        Tscal vsig_B_lambda = [](Tvec v_ab, Tvec r_ab_unit) {
            Tvec v_cross_r = sycl::cross(v_ab, r_ab_unit);
            return sycl::sqrt(
                v_cross_r[0] * v_cross_r[0] + v_cross_r[1] * v_cross_r[1]
                + v_cross_r[2] * v_cross_r[2]);
        };

        inline Tscal
        vsig_hydro(Tscal abs_v_ab_r_ab, Tscal v_A_a, Tscal cs_a, Tscal alpha_av, Tscal beta_av) {
            Tscal v_a  = sycl::sqrt(cs_a * cs_a + v_A_a * v_A_a);
            Tscal vsig = alpha_av * v_a + beta_av * abs_v_ab_r_ab;
            return vsig;
        };

        inline Tscal vsigB(Tvec v_ab, Tvec r_ab_unit) {
            Tvec v_cross_r = sycl::cross(v_ab, r_ab_unit);
            Tscal vsig_B_a = sycl::sqrt(
                v_cross_r[0] * v_cross_r[0] + v_cross_r[1] * v_cross_r[1]
                + v_cross_r[2] * v_cross_r[2]);
            return vsig_B_a;
        };

        inline Tscal vsig_u(Tscal P_a, Tscal P_b, Tscal rho_a, Tscal rho_b) {
            Tscal rho_avg = (rho_a + rho_b) * 0.5;
            Tscal abs_dp  = sham::abs(P_a - P_b);
            return sycl::sqrt(abs_dp / rho_avg);
            // Tscal vsig_u = abs_v_ab_r_ab;
        }

        inline Tscal vsig(
            Tvec v_ab,
            Tvec r_ab_unit,
            Tscal cs_a,
            Tvec B_a,
            Tscal rho_a,
            Tscal mu_0,
            Tscal alpha_av,
            Tscal beta_av) {
            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
            Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);
            Tscal v_a           = v_shock_mhd(cs_a, B_a, rho_a, mu_0);
            Tscal vsig          = alpha_av * v_a + beta_av * abs_v_ab_r_ab;

            return vsig;
        }
    };

} // namespace shamphys
