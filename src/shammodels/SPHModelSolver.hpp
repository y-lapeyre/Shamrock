// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/BasicSPHGhosts.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/sph/SPHUtilities.hpp"
#include <variant>

namespace shammodels {

    template<class Tscal>
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
    };

    template<class Tvec, template<class> class SPHKernel>
    struct SPHModelSolverConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;
        using u_morton           = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        typename InternalEnergyConfig<Tscal>::Variant internal_energy_config;
    };

    /**
     * @brief The shamrock SPH model
     *
     * @tparam Tvec
     * @tparam SPHKernel
     */
    template<class Tvec, template<class> class SPHKernel>
    class SPHModelSolver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;
        using u_morton           = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        using Config = SPHModelSolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        // sph::BasicGas tmp_solver; // temporary all of this should be in the solver in fine

        SPHModelSolver(ShamrockCtx &context)
            : // tmp_solver(context),
              context(context) {}

        Config solver_config;

        static constexpr Tscal htol_up_tol  = 1.2;
        static constexpr Tscal htol_up_iter = 1.2;

        Tscal eos_gamma;
        Tscal gpart_mass;
        Tscal cfl_cour;
        Tscal cfl_force;

        inline void init_required_fields() {
            context.pdata_layout_add_field<Tvec>("xyz", 1);
            context.pdata_layout_add_field<Tvec>("vxyz", 1);
            context.pdata_layout_add_field<Tvec>("axyz", 1);
            context.pdata_layout_add_field<Tscal>("hpart", 1);
            context.pdata_layout_add_field<Tscal>("uint", 1);
            context.pdata_layout_add_field<Tscal>("duint", 1);
        }

        SerialPatchTree<Tvec> gen_serial_patch_tree();

        void apply_position_boundary(SerialPatchTree<Tvec> &sptree);

        Tscal evolve_once(Tscal dt_input,
                          bool enable_physics,
                          bool do_dump,
                          std::string vtk_dump_name,
                          bool vtk_dump_patch_id);
    };

} // namespace shammodels