// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ExtForceConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/legacy/log.hpp"
#include <string>
#include <type_traits>
#include <variant>

namespace shammodels {

    template<class Tvec>
    struct ExtForceConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        struct PointMass {
            Tscal central_mass;
            Tscal Racc;
        };

        struct LenseThirring {
            Tscal central_mass;
            Tscal Racc;
            Tscal a_spin;
            Tvec dir_spin;
        };

        /**
         * \brief Shearing box forces
         * \f[
         *  \partial_t \mathbf{v} + \mathbf{v} \cdot \nabla \mathbf{v} ~ \mapsto \partial_t
         * \mathbf{v} + \mathbf{v} \cdot \nabla \mathbf{v} - \mathbf{f}_{shear} \\
         * \mathbf{f}_{shear} = 2 \eta \, \mathbf{e}_{press} - 2 \Omega (\mathbf{e}_{press} \times
         * \mathbf{e}_{shear}) \times \mathbf{v} \\ \Omega = \frac{2 v_{shear}}{S_{box} s}, \, s =
         * 3/2 \f]
         *
         */
        struct ShearingBoxForce {
            Tscal shear_speed;
            i32_3 shear_base;
            i32_3 shear_dir;

            Tscal pressure_background = 0.01;
            Tscal s                   = 3 / 2;

            inline Tvec get_omega(Tscal box_size) {
                return (2 * shear_speed / (box_size * s)) *
                       sycl::cross(shear_base.convert<Tscal>(), shear_dir.convert<Tscal>());
            }
        };

        using VariantForce = std::variant<PointMass, LenseThirring, ShearingBoxForce>;

        std::vector<VariantForce> ext_forces;

        inline void add_point_mass(Tscal central_mass, Tscal Racc) {
            ext_forces.push_back(PointMass{central_mass, Racc});
        }

        inline void
        add_lense_thirring(Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
            if (sham::abs(sycl::length(dir_spin) - 1) > 1e-8) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the sping direction should be a unit vector");
            }
            ext_forces.push_back(LenseThirring{central_mass, Racc, a_spin, dir_spin});
        }

        /**
         * @brief
         * \todo add check for norm of shear vectors
         */
        inline void add_shearing_box(
            Tscal shear_speed,
            i32_3 shear_base,
            i32_3 shear_dir,
            Tscal pressure_background,
            Tscal s) {

            ext_forces.push_back(
                ShearingBoxForce{shear_speed, shear_base, shear_dir, pressure_background, s});
        }
    };

} // namespace shammodels