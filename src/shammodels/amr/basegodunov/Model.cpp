// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "Model.hpp"

template<class Tvec, class TgridVec>
using Model = shammodels::basegodunov::Model<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Model<Tvec, TgridVec>::init_scheduler(u32 crit_split, u32 crit_merge){

    solver.init_required_fields();
    //solver.init_ghost_layout();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    //sched.add_root_patch();

    //std::cout << "build local" << std::endl;
    //sched.owned_patch_id = sched.patch_list.build_local();
    //sched.patch_list.build_local_idx_map();
    //sched.update_local_dtcnt_value();
    //sched.update_local_load_value();
}


template class shammodels::basegodunov::Model<f64_3, i64_3>;