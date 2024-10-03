// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "config/AVConfig.hpp"
#include "config/BCConfig.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/EOSConfig.hpp"
#include "shammodels/ExtForceConfig.hpp"
#include "shammodels/sph/config/MHDConfig.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <variant>

namespace shammodels::sph {

    /**
     * @brief The configuration for a sph solver
     *
     * @tparam Tvec the type of the vector used to represent the particles
     * @tparam SPHKernel the type of the SPH kernel
     */
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    /**
     * @brief Solver status variables
     *
     * @tparam Tvec the type of the vector used to represent the particles
     */
    template<class Tvec>
    struct SolverStatusVar;

    /**
     * @brief The configuration for the CFL condition
     *
     * @tparam Tscal the type of the scalar used to represent the quantities
     */
    template<class Tscal>
    struct CFLConfig {

        /**
         * @brief The CFL condition for the courant factor
         */
        Tscal cfl_cour;

        /**
         * @brief The CFL condition for the force
         */
        Tscal cfl_force;

        /**
         * @brief The CFL multiplier stiffness
         */
        Tscal cfl_multiplier_stiffness = 2;
    };

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::SolverStatusVar {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time   = 0; ///< Current time
    Tscal dt_sph = 0; ///< Current time step

    Tscal cfl_multiplier = 1e-2; ///< Current cfl multiplier
};

template<class Tvec, template<class> class SPHKernel>
struct shammodels::sph::SolverConfig {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;
    /// The dimension of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    /// The type of the kernel used for the SPH interactions
    using Kernel = SPHKernel<Tscal>;
    /// The type of the Morton code for the tree
    using u_morton = u32;

    /// The radius of the sph kernel
    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass;            ///< The mass of each gas particle
    CFLConfig<Tscal> cfl_config; ///< The configuration for the CFL condition

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// The unit system of the simulation
    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    /// Set the unit system of the simulation
    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }

    /// Retrieves the value of the constant G based on the unit system.
    inline Tscal get_constant_G() {
        if (!unit_sys) {
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }

    /// Retrieves the value of the constant c based on the unit system.
    inline Tscal get_constant_c() {
        if (!unit_sys) {
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.c();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.c();
        }
    }

    inline Tscal get_constant_mu_0() {
        if (!unit_sys) {
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.mu_0();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.mu_0();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to SolverStatusVar type
    using SolverStatusVar = SolverStatusVar<Tvec>;

    /// The time sate of the simulation
    SolverStatusVar time_state;

    /// Set the current time
    inline void set_time(Tscal t) { time_state.time = t; }

    /// Set the time step for the next iteration
    inline void set_next_dt(Tscal dt) { time_state.dt_sph = dt; }

    /// Get the current time
    inline Tscal get_time() { return time_state.time; }

    /// Get the time step for the next iteration
    inline Tscal get_dt_sph() { return time_state.dt_sph; }

    /// Set the CFL multiplier for the time step
    inline void set_cfl_multipler(Tscal lambda) { time_state.cfl_multiplier = lambda; }

    /// Get the CFL multiplier for the time step
    inline Tscal get_cfl_multipler() { return time_state.cfl_multiplier; }

    /// Set the CFL multiplier for the stiffness
    inline void set_cfl_mult_stiffness(Tscal cstiff) {
        cfl_config.cfl_multiplier_stiffness = cstiff;
    }

    /// Get the CFL multiplier for the stiffness
    inline Tscal get_cfl_mult_stiffness() { return cfl_config.cfl_multiplier_stiffness; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using MHDConfig = MHDConfig<Tvec>;
    MHDConfig mhd_type;

    inline void set_noMHD() {
        using Tmp = typename MHDConfig::None;
        mhd_type.set(Tmp{});
    }

    inline void set_IdealMHD(typename MHDConfig::IdealMHD_constrained_hyper_para v) {
        mhd_type.set(v);
    }

    inline void set_NonIdealMHD(typename MHDConfig::NonIdealMHD v) { mhd_type.set(v); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;    ///< Reduction level to be used in the tree build
    bool use_two_stage_search = true; ///< Use two stage neighbors search (see shamrock paper)
    u64 max_neigh_cache_size  = 10e9; ///< Maximum size of the neighbors cache

    /// Setter for the tree reduction level
    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    /// Setter for the two stage search
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }
    /// Setter for the maximum size of the neighbors cache
    inline void set_max_neigh_cache_size(u64 val) { max_neigh_cache_size = val; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool combined_dtdiv_divcurlv_compute = false; ///< Use the combined dtdivv and divcurlv compute
    /// Factor applied to the smoothing lenght for neighbors search (and ghost zone size)
    /// @note This value must be larger or equal to htol_up_iter
    Tscal htol_up_tol = 1.1;
    /// Maximum factor of the smoothing lenght evolution per subcycles
    Tscal htol_up_iter        = 1.1;
    Tscal epsilon_h           = 1e-6; ///< Convergence criteria for the smoothing length
    u32 h_iter_per_subcycles  = 50;   ///< Maximum number of iterations per subcycle
    u32 h_max_subcycles_count = 100;  ///< Maximum number of subcycles before solver crash

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to EOSConfig type
    using EOSConfig = shammodels::EOSConfig<Tvec>;

    /// EOS configuration
    EOSConfig eos_config;

    /// Check if the EOS is a locally isothermal equation of state
    inline bool is_eos_locally_isothermal() {
        using T = typename EOSConfig::LocallyIsothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is an adiabatic equation of state
    inline bool is_eos_adiabatic() {
        using T = typename EOSConfig::Adiabatic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /**
     * @brief Set the EOS configuration to an adiabatic equation of state
     *
     * @param gamma The adiabatic index
     */
    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state
     */
    inline void set_eos_locally_isothermal() { eos_config.set_locally_isothermal(); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state from Lodato
     * Price 2007
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     */
    inline void set_eos_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
        eos_config.set_locally_isothermalLP07(cs0, q, r0);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration for the Artificial Viscosity (AV)
     *
     * @details This struct contains the information needed to configure the Artificial Viscosity
     * in the SPH algorithm. It is a variant of two possible types of artificial viscosity:
     * - None: no AV
     * - Constant: AV with a constant value
     * - VaryingMM97: AV with a varying value, using the Monaghan & Gingold 1997 prescription
     * - VaryingCD10: AV with a varying value, using the Cullen & Dehnen 2010 prescription
     * - ConstantDisc: AV with a constant value, but only in the disc plane
     */
    using AVConfig = AVConfig<Tvec>;

    /// Configuration for the Artificial Viscosity (AV)
    AVConfig artif_viscosity;

    /**
     * @brief Set the artificial viscosity configuration to None
     */
    inline void set_artif_viscosity_None() {
        using Tmp = typename AVConfig::None;
        artif_viscosity.set(Tmp{});
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value
     *
     * @param v Constant value of the artificial viscosity
     */
    inline void set_artif_viscosity_Constant(typename AVConfig::Constant v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Monaghan & Gingold 1997
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingMM97(typename AVConfig::VaryingMM97 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Cullen & Dehnen 2010
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingCD10(typename AVConfig::VaryingCD10 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value in the disc plane.
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) {
        artif_viscosity.set(v);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration of the boundary conditions
     */
    using BCConfig = BCConfig<Tvec>;

    /**
     * @brief Boundary condition configuration
     *
     * See the documentation of the `BCConfig` struct for more informations.
     */
    BCConfig boundary_config;

    /**
     * @brief Set the boundary condition to free boundary
     */
    inline void set_boundary_free() { boundary_config.set_free(); }

    /**
     * @brief Set the boundary condition to periodic boundary
     */
    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    /**
     * @brief Set the boundary condition to shearing periodic boundary
     *
     * The particles are periodic in all directions, but with a shear in the direction
     * given by `shear_dir` and a period of `speed`.
     *
     * @param[in] shear_base The base of the scalar product to define the number of shearing
     * periodicity to be applied
     * @param[in] shear_dir The direction of the shear
     * @param[in] speed The speed of the shear
     */
    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief External force configuration
     *
     * This configuration is used to define the external forces that are applied to the
     * particles in the simulation.
     *
     * The external forces are defined by a variant of different types of forces
     * (i.e., point mass, Lense-Thirring, etc.). The user can add different types
     * of forces using the functions `add_ext_force_point_mass`, `add_ext_force_lense_thirring`,
     * etc.
     */
    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;

    /**
     * @brief External force configuration
     */
    ExtForceConfig ext_force_config{};

    /**
     * @brief Add a point mass external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     */
    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
    }

    /**
     * @brief Add a Lense-Thirring external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     * @param[in] a_spin The spin of the central object
     * @param[in] dir_spin The direction of the spin of the central object
     */
    inline void
    add_ext_force_lense_thirring(Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
        ext_force_config.add_lense_thirring(central_mass, Racc, a_spin, dir_spin);
    }

    /**
     * @brief Add a shearing box external force
     *
     * @param[in] Omega_0 The angular frequency of the shear
     * @param[in] eta The shear rate
     * @param[in] q The power-law index of the shear
     */
    inline void add_ext_force_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {
        ext_force_config.add_shearing_box(Omega_0, eta, q);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether to dump debug information to file
    bool do_debug_dump = false;

    /// @brief The filename to dump debug information in
    std::string debug_dump_filename = "";

    /// @brief Set whether to dump debug information to file
    ///
    /// @param[in] _do_debug_dump Whether to dump debug information to file
    /// @param[in] _debug_dump_filename The filename to dump debug information to
    inline void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename) {
        this->do_debug_dump       = _do_debug_dump;
        this->debug_dump_filename = _debug_dump_filename;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether the ghost cells have a sound speed (i.e. the eos is locally isothermal)
    inline bool ghost_has_soundspeed() { return is_eos_locally_isothermal(); }

    /// @brief Whether the solver has a field for the particle's uint
    ///
    /// @note for now, this is always true as
    inline bool has_field_uint() {
        // no barotropic for now
        return true;
    }

    /// @brief Whether the solver has a field for alpha AV
    inline bool has_field_alphaAV() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for divv
    inline bool has_field_divv() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for dt divv
    inline bool has_field_dtdivv() { return artif_viscosity.has_dtdivv_field(); }

    /// @brief Whether the solver has a field for curlv
    inline bool has_field_curlv() { return artif_viscosity.has_curlv_field() && (dim == 3); }

    /// @brief Whether the solver has a field for B_on_rho
    inline bool has_field_B_on_rho() { return mhd_type.has_B_field() && (dim == 3); }

    /// @brief Whether the solver has a field for psi_on_c
    inline bool has_field_psi_on_ch() { return mhd_type.has_psi_field(); }

    /// @brief Whether the solver has a field for curlB
    inline bool has_field_curlB() { return mhd_type.has_curlB_field() && (dim == 3); }

    /// @brief Whether the solver has a field for ax, ay, az in ghost cells
    inline bool has_axyz_in_ghost() { return has_field_dtdivv(); }

    /// @brief Whether the solver has a field for sound speed
    inline bool has_field_soundspeed() {
        return artif_viscosity.has_field_soundspeed() || is_eos_locally_isothermal();
    }

    /// Print the current status of the solver config
    inline void print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- SPH Solver configuration -----");

        logger::raw_ln("units : ");
        if (unit_sys) {
            logger::raw_ln("unit_length      :", unit_sys->m_inv);
            logger::raw_ln("unit_mass        :", unit_sys->kg_inv);
            logger::raw_ln("unit_current     :", unit_sys->A_inv);
            logger::raw_ln("unit_temperature :", unit_sys->K_inv);
            logger::raw_ln("unit_qte         :", unit_sys->mol_inv);
            logger::raw_ln("unit_lumint      :", unit_sys->cd_inv);
        } else {
            logger::raw_ln("not set");
        }

        logger::raw_ln("part mass", gpart_mass, "( can be changed using .set_part_mass() )");
        logger::raw_ln("cfl force", cfl_config.cfl_force);
        logger::raw_ln("cfl courant", cfl_config.cfl_cour);

        artif_viscosity.print_status();
        eos_config.print_status();
        boundary_config.print_status();

        logger::raw_ln("------------------------------------");
    }
};

namespace shamunits {

    /**
     * @brief Converts a UnitSystem object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The UnitSystem object to be converted.
     */
    template<class Tscal>
    inline void to_json(nlohmann::json &j, const UnitSystem<Tscal> &p) {
        j = nlohmann::json{
            {"unit_time", p.s_inv},
            {"unit_length", p.m_inv},
            {"unit_mass", p.kg_inv},
            {"unit_current", p.A_inv},
            {"unit_temperature", p.K_inv},
            {"unit_qte", p.mol_inv},
            {"unit_lumint", p.cd_inv}};
    }

    /**
     * @brief Deserializes a UnitSystem object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The UnitSystem object to populate.
     */
    template<class Tscal>
    inline void from_json(const nlohmann::json &j, UnitSystem<Tscal> &p) {
        p = UnitSystem<Tscal>(
            j.at("unit_time").get<Tscal>(),
            j.at("unit_length").get<Tscal>(),
            j.at("unit_mass").get<Tscal>(),
            j.at("unit_current").get<Tscal>(),
            j.at("unit_temperature").get<Tscal>(),
            j.at("unit_qte").get<Tscal>(),
            j.at("unit_lumint").get<Tscal>());
    }

} // namespace shamunits

namespace shammodels::sph {

    /**
     * @brief Converts a CFLConfig object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The CFLConfig object to be converted.
     */
    template<class Tscal>
    inline void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
            {"cfl_multiplier_stiffness", p.cfl_multiplier_stiffness}};
    }

    /**
     * @brief Deserializes a CFLConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The CFLConfig object to populate.
     */
    template<class Tscal>
    inline void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to<Tscal>(p.cfl_cour);
        j.at("cfl_force").get_to<Tscal>(p.cfl_force);
        j.at("cfl_multiplier_stiffness").get_to<Tscal>(p.cfl_multiplier_stiffness);
    }

    /**
     * @brief Converts an optional value to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The optional value to be converted.
     *
     * @return None
     *
     * @throws std::bad_optional_access if p is not engaged
     */
    template<class T>
    inline void to_json_optional(nlohmann::json &j, const std::optional<T> &p) {
        if (p) {
            j = *p;
        } else {
            j = {};
        }
    }

    /**
     * @brief Deserializes an optional value from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The optional value to populate.
     *
     * @return None
     *
     * @throws std::bad_optional_access if j is not a valid JSON object
     */
    template<class T>
    inline void from_json_optional(const nlohmann::json &j, std::optional<T> &p) {
        if (j.is_null()) {
            p = std::nullopt;
        } else {
            p = j.get<T>();
        }
    }

    /**
     * @brief Converts a SolverStatusVar object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The SolverStatusVar object to be converted.
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
        j = nlohmann::json{
            {"time", p.time}, {"dt_sph", p.dt_sph}, {"cfl_multiplier", p.cfl_multiplier}};
    }

    /**
     * @brief Deserializes a SolverStatusVar object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverStatusVar object to populate.
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
        using Tscal = typename SolverStatusVar<Tvec>::Tscal;
        j.at("time").get_to<Tscal>(p.time);
        j.at("dt_sph").get_to<Tscal>(p.dt_sph);
        j.at("cfl_multiplier").get_to<Tscal>(p.cfl_multiplier);
    }

    /**
     * @brief Serializes a SolverConfig object to a JSON object.
     *
     * @param j The JSON object to serialize to.
     * @param p The SolverConfig object to serialize.
     */
    template<class Tvec, template<class> class SPHKernel>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        nlohmann::json junit;
        to_json_optional(junit, p.unit_sys);

        j = nlohmann::json{
            // used for type checking
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            // actual data stored in the json
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", junit},
            {"time_state", p.time_state},
            // tree config
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"max_neigh_cache_size", p.max_neigh_cache_size},
            // solver behavior config
            {"combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute},
            {"htol_up_tol", p.htol_up_tol},
            {"htol_up_iter", p.htol_up_iter},
            {"epsilon_h", p.epsilon_h},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},

            {"eos_config", p.eos_config},

            {"artif_viscosity", p.artif_viscosity},
            {"boundary_config", p.boundary_config},
            {"ext_force_config", p.ext_force_config},

            {"do_debug_dump", p.do_debug_dump},
            {"debug_dump_filename", p.debug_dump_filename},
        };
    }

    /**
     * @brief Deserializes a SolverConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverConfig object to populate.
     */
    template<class Tvec, template<class> class SPHKernel>
    inline void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        // type checking
        std::string kernel_id = j.at("kernel_id").get<std::string>();

        if (kernel_id != shambase::get_type_name<Tkernel>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid type to deserialize, wanted " + shambase::get_type_name<Tkernel>()
                + " but got " + kernel_id);
        }

        std::string type_id = j.at("type_id").get<std::string>();

        if (type_id != shambase::get_type_name<Tvec>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                + " but got " + type_id);
        }

        // actual data stored in the json
        j.at("gpart_mass").get_to(p.gpart_mass);
        j.at("cfl_config").get_to(p.cfl_config);

        from_json_optional(j.at("unit_sys"), p.unit_sys);

        j.at("time_state").get_to(p.time_state);

        j.at("tree_reduction_level").get_to(p.tree_reduction_level);
        j.at("use_two_stage_search").get_to(p.use_two_stage_search);
        j.at("max_neigh_cache_size").get_to(p.max_neigh_cache_size);

        j.at("combined_dtdiv_divcurlv_compute").get_to(p.combined_dtdiv_divcurlv_compute);
        j.at("htol_up_tol").get_to(p.htol_up_tol);
        j.at("htol_up_iter").get_to(p.htol_up_iter);
        j.at("epsilon_h").get_to(p.epsilon_h);
        j.at("h_iter_per_subcycles").get_to(p.h_iter_per_subcycles);
        j.at("h_max_subcycles_count").get_to(p.h_max_subcycles_count);

        j.at("eos_config").get_to(p.eos_config);
        j.at("artif_viscosity").get_to(p.artif_viscosity);
        j.at("boundary_config").get_to(p.boundary_config);
        j.at("ext_force_config").get_to(p.ext_force_config);

        j.at("do_debug_dump").get_to(p.do_debug_dump);
        j.at("debug_dump_filename").get_to(p.debug_dump_filename);
    }

} // namespace shammodels::sph
