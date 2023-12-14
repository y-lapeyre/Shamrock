// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::basegodunov::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    StackEntry stack_loc{};

    if(shamcomm::world_rank() == 0){ 
        logger::normal_ln("amr::Godunov", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

    shambase::Timer tstep;
    tstep.start();

    scheduler().update_local_load_value([&](shamrock::patch::Patch p){
        return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    //ghost zone exchange
    modules::GhostZones gz(context,solver_config,storage);
    gz.build_ghost_cache();
#if false
    gz.exchange_ghost();


    //compute bound received
    //round to next pow of 2
    //build radix trees
    modules::AMRTree amrtree(context,solver_config,storage);
    amrtree.build_trees();

    amrtree.correct_bounding_box();
#endif
    
    //compute bound received

    //round to next pow of 2

    //build radix trees

    //build neigh table

    storage.serial_patch_tree.reset();

    return 0;
}


template class shammodels::basegodunov::Solver<f64_3, i64_3>;