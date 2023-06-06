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
#include <memory>
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

        // serial patch tree control
        std::unique_ptr<SerialPatchTree<Tvec>> sptree;
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { sptree.reset(); }

        // interface_control
        std::unique_ptr<sph::BasicSPHGhostHandler<Tvec>> ghost_handler;
        inline void gen_ghost_handler() {
            ghost_handler = std::make_unique<sph::BasicSPHGhostHandler<Tvec>>(scheduler());
        }
        inline void reset_ghost_handler() {

            if (ghost_handler) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "please reset the ghost_handler before");
            }
            ghost_handler.reset();
        }


        







        SPHModelSolver(ShamrockCtx &context) : context(context) {}

        void apply_position_boundary();

        Tscal evolve_once(Tscal dt_input,
                          bool enable_physics,
                          bool do_dump,
                          std::string vtk_dump_name,
                          bool vtk_dump_patch_id);
    };

} // namespace shammodels