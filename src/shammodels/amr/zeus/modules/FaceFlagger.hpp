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


    /**
     * @brief flag faces with a lookup index for the orientation 
     * 
     * 0 = x-
     * 1 = x+
     * 2 = y-
     * 3 = y+
     * 4 = z-
     * 5 = z+
     * @tparam Tvec 
     * @tparam TgridVec 
     */
    template<class Tvec, class TgridVec>
    class FaceFlagger {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        FaceFlagger(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief flag faces with a lookup index 
         * performs at around 2G cell per seconds on a RTX A5000
         */
        void flag_faces();

        inline static Tvec lookup_to_normal(u8 lookup){
            return std::array<Tvec, 6>{
                Tvec{-1,0,0},
                Tvec{ 1,0,0},
                Tvec{0,-1,0},
                Tvec{0, 1,0},
                Tvec{0,0,-1},
                Tvec{0,0, 1},
            }[lookup];
        }

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::zeus::modules