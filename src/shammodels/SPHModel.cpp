// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "SPHModel.hpp"
#include "shamrock/sph/kernels.hpp"

template<class Tvec, template<class> class SPHKernel>
f64 shammodels::SPHModel<Tvec, SPHKernel>::evolve_once(f64 dt_input,
                                                       bool do_dump,
                                                       std::string vtk_dump_name,
                                                       bool vtk_dump_patch_id) {
    return solver.evolve_once(dt_input, do_dump, vtk_dump_name, vtk_dump_patch_id);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::SPHModel<Tvec, SPHKernel>::init_scheduler(u32 crit_split, u32 crit_merge) {
    solver.init_required_fields();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.add_root_patch();

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();
}

template<class Tvec, template<class> class SPHKernel>
u64 shammodels::SPHModel<Tvec, SPHKernel>::get_total_part_count() {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    return shamalgs::collective::allreduce_sum(sched.get_rank_count());
}

template<class Tvec, template<class> class SPHKernel>
f64 shammodels::SPHModel<Tvec, SPHKernel>::total_mass_to_part_mass(f64 totmass) {
    return totmass / get_total_part_count();
}

using namespace shamrock::sph::kernels;

template class shammodels::SPHModel<f64_3, M4>;