// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/AMRTree.hpp"
#include "shammodels/amr/zeus/modules/ComputePressure.hpp"
#include "shammodels/amr/zeus/modules/DiffOperator.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shammodels/amr/zeus/modules/GhostZones.hpp"
#include "shammodels/amr/zeus/modules/SourceStep.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::zeus::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    StackEntry stack_loc{};
    shambase::Timer tstep;
    tstep.start();

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    //ghost zone exchange
    modules::GhostZones gz(context,solver_config,storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();


    //compute bound received
    //round to next pow of 2
    //build radix trees
    modules::AMRTree amrtree(context,solver_config,storage);
    amrtree.build_trees();

    //build neigh table
    amrtree.build_neigh_cache();


    modules::ComputePressure comp_eos(context, solver_config, storage);
    comp_eos.compute_p();
    
    modules::FaceFlagger compute_face_flag(context,solver_config,storage);
    compute_face_flag.flag_faces();
    compute_face_flag.split_face_list();

    
    //modules::DiffOperator diff_op(context,solver_config,storage);
    //diff_op.compute_gradu();

    modules::SourceStep src_step(context,solver_config,storage);
    src_step.compute_forces();
    src_step.apply_force(dt_input);

    storage.face_lists.reset();
    storage.pressure.reset();
    storage.trees.reset();
    storage.merge_patch_bounds.reset();
    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();
    storage.serial_patch_tree.reset();
    
    

    tstep.end();

    u64 rank_count = scheduler().get_rank_count()*AMRBlock::block_size;
    f64 rate = f64(rank_count) / tstep.elasped_sec();

    std::string log_rank_rate = shambase::format(
        "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:3.0f} % |", 
        shamsys::instance::world_rank,rate,  rank_count,  tstep.elasped_sec(),
        100*(storage.timings_details.interface / tstep.elasped_sec()),
        100*(storage.timings_details.neighbors / tstep.elasped_sec()),
        100*(storage.timings_details.io / tstep.elasped_sec())
        );

    std::string gathered = "";
    shamalgs::collective::gather_str(log_rank_rate, gathered);

    if(shamsys::instance::world_rank == 0){
        std::string print = "processing rate infos : \n";
        print+=("---------------------------------------------------------------------------------\n");
        print+=("| rank |  rate  (N.s^-1)  |      N      | t compute (s) | interf | neigh |   io  |\n");
        print+=("---------------------------------------------------------------------------------");
        print+=(gathered) + "\n";
        print+=("---------------------------------------------------------------------------------");
        logger::info_ln("amr::Zeus",print);
    }

    storage.timings_details.reset();

    return 0;
}


template class shammodels::zeus::Solver<f64_3, i64_3>;