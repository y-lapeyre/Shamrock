// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::substep_1(){


    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf      = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf     = ghost_layout.get_field_idx<Tscal>("eint");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData & pdat){

        MergedPDat & mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            

        });

    });

}


template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;