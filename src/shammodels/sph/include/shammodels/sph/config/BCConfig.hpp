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
 * @file BCConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph {

    /**
     * @brief Boundary conditions configuration
     *
     * This struct is used to configure the boundary conditions of a simulation.
     *
     * @tparam Tvec The vector type used for the simulation.
     */
    template<class Tvec>
    struct BCConfig;

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::BCConfig {

    /// Type of the components of the vector of coordinates
    using Tscal = shambase::VecComponent<Tvec>;
    /// Number of dimensions of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief Free boundary condition
     *
     * The box will be expanded if a particle is outside of it.
     */
    struct Free {
        /**
         * @brief The tolerance for the box expansion
         *
         * If a particle is outside of the box, the box will be expanded to the new range with an
         * added margin factor of expand_tolerance
         */
        Tscal expand_tolerance = 1.2;
    };

    /**
     * @brief Periodic boundary condition
     */
    struct Periodic {};

    /**
     * @brief Shearing periodic boundary condition
     * @todo use a bib entry instead
     * @see https://ui.adsabs.harvard.edu/abs/2010ApJS..189..142S/abstract
     */

    struct ShearingPeriodic {
        /**
         * @brief The base of the scalar product to define the number of shearing periodicity to be
         * applied
         */
        i32_3 shear_base;

        /**
         * @brief The direction of the shear
         */
        i32_3 shear_dir;

        /**
         * @brief The speed of the shear
         */
        Tscal shear_speed;
    };

    struct HardWall {
        /**
         * @brief Number of layers of ghost particles to create
         * For hard walls, typically you want THICK walls
         */
        u32 ghost_layers = 2;

        /**
         * @brief Damping factor for wall interactions
         * Higher values mean more damping (0.0 = no damping, 1.0 = full damping)
         */
        Tscal damping_factor = 0.5;

        /**
         * @brief Whether to apply repulsive force at walls
         */
        bool use_repulsive_force = true;

        /**
         * @brief Repulsive force coefficient
         */
        Tscal repulsive_coefficient = 100.0;

        /**
         * @brief Distance from wall where repulsive force starts
         */
        Tscal repulsive_range = 0.1;
    };

    /// Variant of all types of artificial viscosity possible
    using Variant = std::variant<Free, Periodic, ShearingPeriodic, HardWall>;

    /// The actual configuration (default to free boundaries)
    Variant config = Free{};

    /// Set the boundary condition to free boundaries
    inline void set_free() { config = Free{}; }

    /// Set the boundary condition to periodic boundaries
    inline void set_periodic() { config = Periodic{}; }

    /**
     * @brief Set the boundary condition to shearing periodic boundaries
     *
     * @param shear_base The base of the scalar product to define the number of shearing periodicity
     * to be applied
     * @param shear_dir The direction of the shear
     * @param speed The speed of the shear
     */
    inline void set_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        config = ShearingPeriodic{shear_base, shear_dir, speed};
    }

    /**
     * @brief Set the boundary condition to hardwall boundaries
     *
     * @param ghost_layers Number of layers of ghost particles to create
     * @param damping_factor Damping factor for wall interactions
     * @param use_repulsive_force Whether to apply repulsive force at walls
     * @param repulsive_coefficient Repulsive force coefficient
     * @param repulsive_range Distance from wall where repulsive force starts
     */
    inline void set_hard_wall(
        u32 ghost_layers,
        Tscal damping_factor,
        bool use_repulsive_force,
        Tscal repulsive_coefficient,
        Tscal repulsive_range) {
        config = HardWall{
            ghost_layers,
            damping_factor,
            use_repulsive_force,
            repulsive_coefficient,
            repulsive_range};
    }

    /**
     * @brief Prints the current boundary condition configuration to the logger.
     *
     * The function logs the type of boundary condition (free, periodic, or shearing periodic)
     * and the relevant parameters for the shearing periodic case.
     */
    inline void print_status() {
        logger::raw_ln("--- Bondaries config");

        if (Free *v = std::get_if<Free>(&config)) {
            logger::raw_ln("  Config Type : Free boundaries");
        } else if (Periodic *v = std::get_if<Periodic>(&config)) {
            logger::raw_ln("  Config Type : Periodic boundaries");
        } else if (ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&config)) {
            logger::raw_ln("  Config Type : ShearingPeriodic (Stone 2010)");
            logger::raw_ln("  shear_base   =", v->shear_base);
            logger::raw_ln("  shear_dir   =", v->shear_dir);
            logger::raw_ln("  shear_speed =", v->shear_speed);
        } else if (HardWall *v = std::get_if<HardWall>(&config)) {
            logger::raw_ln("  Config Type : HardWall");
            logger::raw_ln("  ghost_layers   =", v->ghost_layers);
            logger::raw_ln("  damping_factor   =", v->damping_factor);
            logger::raw_ln("  use_repulsive_force   =", v->use_repulsive_force);
            logger::raw_ln("  repulsive_coefficient   =", v->repulsive_coefficient);
            logger::raw_ln("  repulsive_range   =", v->repulsive_range);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- Bondaries config config (deduced)");

        logger::raw_ln("-------------");
    }
};

namespace shammodels::sph {

    /**
     * @brief Serialize a BCConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The BCConfig to serialize
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;
        using HardWall         = typename T::HardWall;

        // Write the config type into the JSON object
        if (const Free *v = std::get_if<Free>(&p.config)) {
            j = {
                {"bc_type", "free"},
            };
        } else if (const Periodic *v = std::get_if<Periodic>(&p.config)) {
            j = {
                {"bc_type", "periodic"},
            };
        } else if (const ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&p.config)) {
            // Write the shear base, direction, and speed into the JSON object
            j = {
                {"bc_type", "shearing_periodic"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"shear_speed", v->shear_speed},
            };
        } else if (const HardWall *v = std::get_if<HardWall>(&p.config)) {
            j = {
                {"bc_type", "hardwall"},
                {"ghost_layers", v->ghost_layers},
                {"damping_factor", v->damping_factor},
                {"use_repulsive_force", v->use_repulsive_force},
                {"repulsive_coefficient", v->repulsive_coefficient},
                {"repulsive_range", v->repulsive_range},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    /**
     * @brief Deserialize a JSON object into a BCConfig
     *
     * @param[in] j  The JSON object to read from
     * @param[out] p The BCConfig to deserialize
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        // Check if the JSON object contains the "bc_type" field
        if (!j.contains("bc_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        // Read the config type from the JSON object
        std::string bc_type;
        j.at("bc_type").get_to(bc_type);

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;
        using HardWall         = typename T::HardWall;

        // Set the BCConfig based on the config type
        if (bc_type == "free") {
            p.set_free();
        } else if (bc_type == "periodic") {
            p.set_periodic();
        } else if (bc_type == "shearing_periodic") {
            p.set_shearing_periodic(
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("speed").get<Tscal>());
        } else if (bc_type == "hardwall") {
            p.set_hardwall(
                j.at("ghost_layers").get<u32>(),
                j.at("damping_factor").get<Tscal>(),
                j.at("use_repulsive_force").get<bool>(),
                j.at("repulsive_coefficient").get<Tscal>(),
                j.at("repulsive_range").get<Tscal>());
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

} // namespace shammodels::sph
