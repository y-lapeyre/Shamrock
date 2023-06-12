// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/legacy/log.hpp"
#include <variant>

namespace shammodels {
    template<class Tvec, template<class> class SPHKernel>
    struct SPHModelSolverConfig;
}

template<class Tvec, template<class> class SPHKernel>
struct shammodels::SPHModelSolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    static constexpr Tscal Rkern = Kernel::Rkern;

    struct InternalEnergyConfig {
        struct None {};

        /**
         * @brief cf Price 2018 , q^a_ab = 0
         */
        struct NoAV {};

        struct ConstantAv {
            Tscal alpha_u  = 1.0;
            Tscal alpha_AV = 1.0;
            Tscal beta_AV  = 2.0;
        };
        struct VaryingAv {
            Tscal sigma_decay = 0.1;
            Tscal alpha_u     = 1.0;
            Tscal beta_AV     = 2.0;
        };

        using Variant  = std::variant<None, NoAV, ConstantAv, VaryingAv>;
        Variant config = None{};

        void set(Variant v) { config = v; }

        inline bool has_uint_field() {
            bool is_none = std::get_if<None>(&config);
            return !is_none;
        }

        inline bool has_alphaAV_field() {
            bool is_varying_alpha = std::get_if<VaryingAv>(&config);
            return is_varying_alpha;
        }

        inline void print_status() {
            logger::raw_ln("--- internal energy config");

            if (None *v = std::get_if<None>(&config)) {
                logger::raw_ln("Config Type : None");
            } else if (NoAV *v = std::get_if<NoAV>(&config)) {
                logger::raw_ln("Config Type : NoAV (No artificial viscosity)");
            } else if (ConstantAv *v = std::get_if<ConstantAv>(&config)) {
                logger::raw_ln("Config Type : ConstantAv (Constant artificial viscosity)");
                logger::raw_ln("alpha_u  =", v->alpha_u);
                logger::raw_ln("alpha_AV =", v->alpha_AV);
                logger::raw_ln("beta_AV  =", v->beta_AV);
            } else if (VaryingAv *v = std::get_if<VaryingAv>(&config)) {
                logger::raw_ln("Config Type : VaryingAv (Varying artificial viscosity)");
                logger::raw_ln("sigma_decay =", v->sigma_decay);
                logger::raw_ln("alpha_u     =", v->alpha_u);
                logger::raw_ln("beta_AV     =", v->beta_AV);
            }

            logger::raw_ln("--- internal energy config (deduced)");

            logger::raw_ln("-------------");
        }
    };

    InternalEnergyConfig internal_energy_config;

    inline void set_internal_energy_config_none() {
        using Tmp = typename InternalEnergyConfig::None;
        internal_energy_config.set(Tmp{});
    }

    inline void set_internal_energy_config_NoAV() {
        using Tmp = typename InternalEnergyConfig::NoAV;
        internal_energy_config.set(Tmp{});
    }

    inline void set_internal_energy_config_ConstantAv(typename InternalEnergyConfig::ConstantAv v) {
        internal_energy_config.set(v);
    }

    inline void set_internal_energy_config_VaryingAv(typename InternalEnergyConfig::VaryingAv v) {
        internal_energy_config.set(v);
    }

    inline bool has_uint_field() { return internal_energy_config.has_uint_field(); }

    inline bool has_alphaAV_field() { return internal_energy_config.has_alphaAV_field(); }

    inline bool has_divv_field() { return internal_energy_config.has_alphaAV_field(); }
    inline bool has_curlv_field() {
        return internal_energy_config.has_alphaAV_field() && (dim == 3);
    }

    inline void print_status() { internal_energy_config.print_status(); }
};
