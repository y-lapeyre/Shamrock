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
#include "shamcomm/collectives.hpp"
#include "shammodels/amr/basegodunov/modules/AMRGraphGen.hpp"
#include "shammodels/amr/basegodunov/modules/AMRTree.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeCellInfos.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeFlux.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeGradient.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeTimeDerivative.hpp"
#include "shammodels/amr/basegodunov/modules/FaceInterpolate.hpp"
#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"
#include "shammodels/amr/basegodunov/modules/StencilGenerator.hpp"

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

    gz.exchange_ghost();

    modules::ComputeCellInfos comp_cell_infos(context,solver_config,storage);
    comp_cell_infos.compute_aabb();


    //compute bound received
    //round to next pow of 2
    //build radix trees
    modules::AMRTree amrtree(context,solver_config,storage);
    amrtree.build_trees();

    amrtree.correct_bounding_box();

    

    //modules::StencilGenerator stencil_gen(context,solver_config,storage);
    //stencil_gen.make_stencil();

    modules::AMRGraphGen graph_gen(context,solver_config,storage);
    auto block_oriented_graph = graph_gen.find_AMR_block_graph_links_common_face();

    graph_gen.lower_AMR_block_graph_to_cell_common_face_graph(block_oriented_graph);
    
    // compute & limit gradients
    modules::ComputeGradient grad_compute(context,solver_config,storage);
    grad_compute.compute_grad_rho_van_leer();
    grad_compute.compute_grad_rhov_van_leer();
    grad_compute.compute_grad_rhoe_van_leer();

    // shift values
    modules::FaceInterpolate face_interpolator(context,solver_config,storage);
    face_interpolator.interpolate_rho_to_face();
    face_interpolator.interpolate_rhov_to_face();
    face_interpolator.interpolate_rhoe_to_face();

    // flux
    modules::ComputeFlux flux_compute(context,solver_config,storage);
    flux_compute.compute_flux_rusanov();

    //compute dt fields
    modules::ComputeTimeDerivative dt_compute(context,solver_config,storage);
    dt_compute.compute_dt_fields();

    // RK2 + flux lim



    storage.cell_link_graph.reset();
    storage.serial_patch_tree.reset();




    tstep.end();

    u64 rank_count = scheduler().get_rank_count()*AMRBlock::block_size;
    f64 rate = f64(rank_count) / tstep.elasped_sec();

    std::string log_rank_rate = shambase::format(
        "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:3.0f} % |", 
        shamcomm::world_rank(),rate,  rank_count,  tstep.elasped_sec(),
        100*(storage.timings_details.interface / tstep.elasped_sec()),
        100*(storage.timings_details.neighbors / tstep.elasped_sec()),
        100*(storage.timings_details.io / tstep.elasped_sec())
        );

    std::string gathered = "";
    shamcomm::gather_str(log_rank_rate, gathered);

    if(shamcomm::world_rank() == 0){
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


template class shammodels::basegodunov::Solver<f64_3, i64_3>;