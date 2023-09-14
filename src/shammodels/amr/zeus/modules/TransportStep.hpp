// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    template<class Tvec, class TgridVec>
    class TransportStep {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec,TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        TransportStep(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Compute face momentas
         * (\cite Fargo3D_1 eq 48 49)
         * 
         * \f{eqnarray*}{
         *     \Pi^{-x}_{i,j,k} &=& \rho_i v_{i,j,k}\\ 
         *     \Pi^{+x}_{i,j,k} &=& \rho_i v_{i+1,j,k}
         * \f}
         * same goes for y,z
         */
        void compute_cell_centered_momentas();

        void compute_limiter();

        void compute_face_centered_moments(Tscal dt);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

    };

} // namespace shammodels::zeus::modules