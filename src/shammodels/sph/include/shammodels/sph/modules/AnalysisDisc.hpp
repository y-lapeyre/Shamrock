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
 * @file AnalysisDisc.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamphys/SodTube.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisDisc {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        shamphys::SodTube solution;
        Tvec direction;
        Tscal time_val;
        Tscal x_ref;        // shock centered on x_ref
        Tscal x_min, x_max; // check only between [x_min, x_max ]

        AnalysisDisc(
            ShamrockCtx &context,
            Config &solver_config,
            Storage &storage,
            shamphys::SodTube &solution,
            Tvec direction,
            Tscal time_val,
            Tscal x_ref,
            Tscal x_min,
            Tscal x_max)
            : context(context), solver_config(solver_config), storage(storage), solution(solution),
              direction(direction), time_val(time_val), x_ref(x_ref), x_min(x_min), x_max(x_max) {}

        struct field_val {
            Tscal rho;
            Tvec v;
            Tscal P;
        };

        field_val compute_L2_dist();

        struct analysis_val {
            int ibin;
            Tscal radius;
            int counter;
            Tvec J;
            Tscal Sigma;
            Tvec l;
            Tscal tilt;
            Tscal twist;
            Tscal psi;
            Tscal H_on_r;
        };

        struct analysis_basis {

            sycl::buffer<Tscal> lx;
            sycl::buffer<Tscal> ly;
            sycl::buffer<Tscal> lz;
            sycl::buffer<Tscal> Sigma;
        };

        struct analysis_stage1 {

            sycl::buffer<Tscal> tilt;
            sycl::buffer<Tscal> twist;
            sycl::buffer<Tscal> psi;
            sycl::buffer<Tscal> zmean;
        };

        struct analysis_stage2 {

            sycl::buffer<Tscal> H;
            sycl::buffer<Tscal> H_on_R;
        };

        analysis_val compute_analysis();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
