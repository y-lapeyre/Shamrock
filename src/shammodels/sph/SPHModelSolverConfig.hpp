// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include <variant>

namespace shammodels {

    template<class Tvec, template<class> class SPHKernel>
    struct SPHModelSolverConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;
        using u_morton           = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        struct InternalEnergyConfig {
            struct None {};
            struct NoAV {};
            struct ConstantAv {
                Tscal alpha_u  = 1.0;
                Tscal alpha_AV = 1.0;
                Tscal beta_AV  = 2.0;
            };
            struct VaryingAv {
                Tscal sigma_decay = 0.1;
                Tscal alpha_u     = 1.0;
            };

            using Variant = std::variant<None, NoAV, ConstantAv, VaryingAv>;

            inline static bool has_uint_field(Variant &v) {
                bool is_none = std::get_if<None>(&v);
                return !is_none;
            }

            inline static bool has_alphaAV_field(Variant &v) {
                bool is_varying_alpha = std::get_if<VaryingAv>(&v);
            }
        };

        typename InternalEnergyConfig::Variant internal_energy_config;

        inline bool has_uint_field() {
            return InternalEnergyConfig::has_uint_field(internal_energy_config);
        }
        inline bool has_alphaAV_field() {
            return InternalEnergyConfig::has_alphaAV_field(internal_energy_config);
        }
    };

} // namespace shammodels