// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SPHSetup.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class SPHSetup {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SPHSetup(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void apply_setup(
            SetupNodePtr setup,
            bool part_reordering,
            std::optional<u32> insert_step = std::nullopt);

        std::shared_ptr<ISPHSetupNode>
        make_generator_lattice_hcp(Tscal dr, std::pair<Tvec, Tvec> box);

        std::shared_ptr<ISPHSetupNode> make_generator_disc_mc(
            Tscal part_mass,
            Tscal disc_mass,
            Tscal r_in,
            Tscal r_out,
            std::function<Tscal(Tscal)> sigma_profile,
            std::function<Tscal(Tscal)> H_profile,
            std::function<Tscal(Tscal)> rot_profile,
            std::function<Tscal(Tscal)> cs_profile,
            std::mt19937 eng);

        std::shared_ptr<ISPHSetupNode>
        make_combiner_add(SetupNodePtr parent1, SetupNodePtr parent2);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
