// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/AMRGodunovSolver.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
namespace shammodels {

    template<class Tvec, class TgridVec>
    class AMRGodunovModel {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        ShamrockCtx &ctx;

        using Solver = AMRGodunovSolver<Tvec, TgridVec>;
        Solver solver;

        AMRGodunovModel(ShamrockCtx &ctx) : ctx(ctx), solver(ctx){};
    };

} // namespace shammodels