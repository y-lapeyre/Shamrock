// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shammodels/BasicSPHGhosts.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"

namespace shammodels::sph {

    /**
     * @brief handle basic utilities dealing with SPH
     *
     * @tparam vec
     */
    template<class vec, class SPHKernel>
    class SPHHandler {
        public:
        using flt = shambase::VecComponent<vec>;

        static constexpr flt Rkern = SPHKernel::Rkern;

        PatchScheduler &sched;
        flt h_evol_max;
        BasicSPHGhostHandler<vec> &interf_handle;

        SPHHandler(PatchScheduler &sched, BasicSPHGhostHandler<vec> &interf, flt h_evol_max)
            : sched(sched), h_evol_max(h_evol_max), interf_handle(interf) {}

        using GhostHndl = BasicSPHGhostHandler<vec>;
        using InterfBuildCache =
            shambase::DistributedDataShared<typename GhostHndl::InterfaceIdTable>;

        inline InterfBuildCache build_interf_cache(SerialPatchTree<vec> &sptree) {

            using namespace shamrock::patch;

            const u32 ihpart = sched.pdl.get_field_idx<flt>("hpart");

            PatchField<flt> interactR_patch = sched.map_owned_to_patch_field_simple<flt>(
                [&](const Patch p, PatchData &pdat) -> flt {
                    if (!pdat.is_empty()) {
                        return pdat.get_field<flt>(ihpart).compute_max() * h_evol_max * Rkern;
                    } else {
                        return shambase::VectorProperties<flt>::get_min();
                    }
                });

            PatchtreeField<flt> interactR_mpi_tree = sptree.make_patch_tree_field(
                sched,
                shamsys::instance::get_compute_queue(),
                interactR_patch,
                [](flt h0, flt h1, flt h2, flt h3, flt h4, flt h5, flt h6, flt h7) {
                    return shambase::sycl_utils::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
                });

            return interf_handle.make_interface_cache(sptree, interactR_mpi_tree, interactR_patch);
        }
    };

} // namespace shammodels::sph