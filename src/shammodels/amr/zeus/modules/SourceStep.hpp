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
    class SourceStep {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec,TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SourceStep(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief compute general forces (pressure + external and store them into `SolverStorage::forces`)
         * \f[
         *   \frac{u_i^{n+1} - u_i^{n}}{\Delta t} = \underbrace{-\frac{\partial_i p^n}{\rho^n} 
         *     + f_{\text{ext},i}}_{ f_{\rm gen} }
         * \f]
         */
        void compute_forces();

        /**
         * @brief Compute the values of the artificial viscosity terms
         * (\cite Zeus2d_main equations 33,34)
         * 
         * \f{eqnarray*}{
         *     q^x_{i,j} &=& C^{\rm AV} \rho_{i,j} (v^x_{i+1,j} - v^x_{i,j}) \\ 
         *     q^y_{i,j} &=& C^{\rm AV} \rho_{i,j} (v^y_{i,j+1} - v^y_{i,j})
         * \f}
         */
        void compute_AV();

        /**
         * @brief apply the generalized forces
         * \f[
         *   \frac{u_i^{n+1} - u_i^{n}}{\Delta t} = \underbrace{-\frac{\partial_i p^n}{\rho^n} 
         *     + f_{\text{ext},i}}_{ f_{\rm gen} }
         * \f]
         */
        void apply_force(Tscal dt);
        void substep_2();
        void substep_3();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::zeus::modules