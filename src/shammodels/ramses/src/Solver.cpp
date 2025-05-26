// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/Solver.hpp"
#include "shamcomm/collectives.hpp"
#include "shammodels/common/timestep_report.hpp"
#include "shammodels/ramses/modules/AMRGraphGen.hpp"
#include "shammodels/ramses/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/ramses/modules/AMRTree.hpp"
#include "shammodels/ramses/modules/ComputeCFL.hpp"
#include "shammodels/ramses/modules/ComputeCellInfos.hpp"
#include "shammodels/ramses/modules/ComputeFlux.hpp"
#include "shammodels/ramses/modules/ComputeGradient.hpp"
#include "shammodels/ramses/modules/ComputeTimeDerivative.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shammodels/ramses/modules/FaceInterpolate.hpp"
#include "shammodels/ramses/modules/GhostZones.hpp"
#include "shammodels/ramses/modules/StencilGenerator.hpp"
#include "shammodels/ramses/modules/TimeIntegrator.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::evolve_once() {

    StackEntry stack_loc{};

    sham::MemPerfInfos mem_perf_infos_start = sham::details::get_mem_perf_info();

    Tscal t_current = solver_config.get_time();
    Tscal dt_input  = solver_config.get_dt();

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("amr::Godunov", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

    shambase::Timer tstep;
    tstep.start();

    // Scheduler step
    auto update_load_val = [&]() {
        logger::debug_ln("ComputeLoadBalanceValue", "update load balancing");
        scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
            return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
        });
    };
    update_load_val();
    scheduler().scheduler_step(true, true);
    update_load_val();
    scheduler().scheduler_step(false, false);

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    // ghost zone exchange
    modules::GhostZones gz(context, solver_config, storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();

    modules::ComputeCellInfos comp_cell_infos(context, solver_config, storage);
    comp_cell_infos.compute_aabb();

    // compute bound received
    // round to next pow of 2
    // build radix trees
    modules::AMRTree amrtree(context, solver_config, storage);
    amrtree.build_trees();

    amrtree.correct_bounding_box();

    // modules::StencilGenerator stencil_gen(context,solver_config,storage);
    // stencil_gen.make_stencil();

    modules::AMRGraphGen graph_gen(context, solver_config, storage);
    auto block_oriented_graph = graph_gen.find_AMR_block_graph_links_common_face();

    graph_gen.lower_AMR_block_graph_to_cell_common_face_graph(block_oriented_graph);

    // compute prim variable
    modules::ConsToPrim ctop(context, solver_config, storage);

    ctop.cons_to_prim_gas();
    if (solver_config.is_dust_on()) {
        ctop.cons_to_prim_dust();
    }

    // compute & limit gradients
    modules::ComputeGradient grad_compute(context, solver_config, storage);
    grad_compute.compute_grad_rho_van_leer();
    grad_compute.compute_grad_v_van_leer();
    grad_compute.compute_grad_P_van_leer();
    if (solver_config.is_dust_on()) {
        grad_compute.compute_grad_rho_dust_van_leer();
        grad_compute.compute_grad_v_dust_van_leer();
    }

    // shift values
    modules::FaceInterpolate face_interpolator(context, solver_config, storage);
    Tscal dt_face_interp = 0;
    if (solver_config.face_half_time_interpolation) {
        dt_face_interp = dt_input / 2.0;
    }
    face_interpolator.interpolate_rho_to_face(dt_face_interp);
    face_interpolator.interpolate_v_to_face(dt_face_interp);
    face_interpolator.interpolate_P_to_face(dt_face_interp);

    if (solver_config.is_dust_on()) {
        face_interpolator.interpolate_rho_dust_to_face(dt_face_interp);
        face_interpolator.interpolate_v_dust_to_face(dt_face_interp);
    }

    // flux
    modules::ComputeFlux flux_compute(context, solver_config, storage);
    flux_compute.compute_flux();
    if (solver_config.is_dust_on()) {
        flux_compute.compute_flux_dust();
    }
    // compute dt fields
    modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
    dt_compute.compute_dt_fields();
    if (solver_config.is_dust_on()) {
        dt_compute.compute_dt_dust_fields();
    }

    // RK2 + flux lim
    if (solver_config.drag_config.drag_solver_config == DragSolverMode::NoDrag) {
        modules::TimeIntegrator dt_integ(context, solver_config, storage);
        dt_integ.forward_euler(dt_input);
    } else if (solver_config.drag_config.drag_solver_config == DragSolverMode::IRK1) {
        modules::DragIntegrator drag_integ(context, solver_config, storage);
        drag_integ.involve_with_no_src(dt_input);
        drag_integ.enable_irk1_drag_integrator(dt_input);
    } else {
        shambase::throw_unimplemented();
    }

    modules::AMRGridRefinementHandler refinement(context, solver_config, storage);
    refinement.update_refinement();

    modules::ComputeCFL cfl_compute(context, solver_config, storage);
    f64 new_dt = cfl_compute.compute_cfl();

    // if new physics like dust is added then use the smallest dt
    if (solver_config.is_dust_on())
        new_dt = std::min(new_dt, cfl_compute.compute_dust_cfl());

    solver_config.set_next_dt(new_dt);
    solver_config.set_time(t_current + dt_input);

    storage.dtrho.reset();
    storage.dtrhov.reset();
    storage.dtrhoe.reset();

    storage.flux_rho_face_xp.reset();
    storage.flux_rho_face_xm.reset();
    storage.flux_rho_face_yp.reset();
    storage.flux_rho_face_ym.reset();
    storage.flux_rho_face_zp.reset();
    storage.flux_rho_face_zm.reset();
    storage.flux_rhov_face_xp.reset();
    storage.flux_rhov_face_xm.reset();
    storage.flux_rhov_face_yp.reset();
    storage.flux_rhov_face_ym.reset();
    storage.flux_rhov_face_zp.reset();
    storage.flux_rhov_face_zm.reset();
    storage.flux_rhoe_face_xp.reset();
    storage.flux_rhoe_face_xm.reset();
    storage.flux_rhoe_face_yp.reset();
    storage.flux_rhoe_face_ym.reset();
    storage.flux_rhoe_face_zp.reset();
    storage.flux_rhoe_face_zm.reset();

    storage.rho_face_xp.reset();
    storage.rho_face_xm.reset();
    storage.rho_face_yp.reset();
    storage.rho_face_ym.reset();
    storage.rho_face_zp.reset();
    storage.rho_face_zm.reset();

    storage.vel_face_xp.reset();
    storage.vel_face_xm.reset();
    storage.vel_face_yp.reset();
    storage.vel_face_ym.reset();
    storage.vel_face_zp.reset();
    storage.vel_face_zm.reset();

    storage.press_face_xp.reset();
    storage.press_face_xm.reset();
    storage.press_face_yp.reset();
    storage.press_face_ym.reset();
    storage.press_face_zp.reset();
    storage.press_face_zm.reset();

    storage.grad_rho.reset();
    storage.dx_v.reset();
    storage.dy_v.reset();
    storage.dz_v.reset();
    storage.grad_P.reset();

    ctop.reset_gas();

    if (solver_config.is_dust_on()) {
        storage.dtrho_dust.reset();
        storage.dtrhov_dust.reset();

        storage.flux_rho_dust_face_xp.reset();
        storage.flux_rho_dust_face_xm.reset();
        storage.flux_rho_dust_face_yp.reset();
        storage.flux_rho_dust_face_ym.reset();
        storage.flux_rho_dust_face_zp.reset();
        storage.flux_rho_dust_face_zm.reset();
        storage.flux_rhov_dust_face_xp.reset();
        storage.flux_rhov_dust_face_xm.reset();
        storage.flux_rhov_dust_face_yp.reset();
        storage.flux_rhov_dust_face_ym.reset();
        storage.flux_rhov_dust_face_zp.reset();
        storage.flux_rhov_dust_face_zm.reset();

        storage.rho_dust_face_xm.reset();
        storage.rho_dust_face_yp.reset();
        storage.rho_dust_face_ym.reset();
        storage.rho_dust_face_xp.reset();
        storage.rho_dust_face_zp.reset();
        storage.rho_dust_face_zm.reset();

        storage.vel_dust_face_xp.reset();
        storage.vel_dust_face_xm.reset();
        storage.vel_dust_face_yp.reset();
        storage.vel_dust_face_ym.reset();
        storage.vel_dust_face_zp.reset();
        storage.vel_dust_face_zm.reset();

        storage.grad_rho_dust.reset();
        storage.dx_v_dust.reset();
        storage.dy_v_dust.reset();
        storage.dz_v_dust.reset();

        ctop.reset_dust();
    }

    if (solver_config.drag_config.drag_solver_config != DragSolverMode::NoDrag) {
        storage.rho_next_no_drag.reset();
        storage.rhov_next_no_drag.reset();
        storage.rhoe_next_no_drag.reset();
        storage.rho_d_next_no_drag.reset();
        storage.rhov_d_next_no_drag.reset();
    }

    storage.cell_infos.reset();
    storage.cell_link_graph.reset();

    storage.trees.reset();
    storage.merge_patch_bounds.reset();

    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();

    storage.serial_patch_tree.reset();

    tstep.end();

    sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

    f64 t_dev_alloc
        = (mem_perf_infos_end.time_alloc_device - mem_perf_infos_start.time_alloc_device)
          + (mem_perf_infos_end.time_free_device - mem_perf_infos_start.time_free_device);

    u64 rank_count = scheduler().get_rank_count() * AMRBlock::block_size;
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

    std::string log_step = report_perf_timestep(
        rate,
        rank_count,
        tstep.elasped_sec(),
        storage.timings_details.interface,
        t_dev_alloc,
        mem_perf_infos_end.max_allocated_byte_device);
        
    t_tot_cumul +=    shamalgs::collective::allreduce_max(tstep.elasped_sec());


    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::RAMSES", log_step);
        logger::info_ln(
            "amr::RAMSES",
            "estimated rate :",
            dt_input * (3600 / tstep.elasped_sec()),
            "(tsim/hr)");
            std::cout << "\n    curent    cumul time (s)     " << t_tot_cumul << "\n\n";
    }

    storage.timings_details.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::do_debug_vtk_dump(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    u32 block_size = Solver::AMRBlock::block_size;

    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
    std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

    sycl::buffer<Tvec> pos_min_cell(num_obj * block_size);
    sycl::buffer<Tvec> pos_max_cell(num_obj * block_size);

    shamsys::instance::get_compute_queue().submit([&, block_size](sycl::handler &cgh) {
        sycl::accessor acc_p1{shambase::get_check_ref(pos1), cgh, sycl::read_only};
        sycl::accessor acc_p2{shambase::get_check_ref(pos2), cgh, sycl::read_only};
        sycl::accessor cell_min{pos_min_cell, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor cell_max{pos_max_cell, cgh, sycl::write_only, sycl::no_init};

        using Block = typename Solver::AMRBlock;

        shambase::parralel_for(cgh, num_obj, "rescale cells", [=](u64 id_a) {
            Tvec block_min = acc_p1[id_a].template convert<Tscal>();
            Tvec block_max = acc_p2[id_a].template convert<Tscal>();

            Tvec delta_cell = (block_max - block_min) / Block::side_size;
#pragma unroll
            for (u32 ix = 0; ix < Block::side_size; ix++) {
#pragma unroll
                for (u32 iy = 0; iy < Block::side_size; iy++) {
#pragma unroll
                    for (u32 iz = 0; iz < Block::side_size; iz++) {
                        u32 i                           = Block::get_index({ix, iy, iz});
                        Tvec delta_val                  = delta_cell * Tvec{ix, iy, iz};
                        cell_min[id_a * block_size + i] = block_min + delta_val;
                        cell_max[id_a * block_size + i] = block_min + (delta_cell) + delta_val;
                    }
                }
            }
        });
    });

    writer.write_voxel_cells(pos_min_cell, pos_max_cell, num_obj * block_size);

    writer.add_cell_data_section();
    writer.add_field_data_section(11);

    std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", fields_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(3);
    writer.write_field("rhovel", fields_vel, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(4);
    writer.write_field("rhoetot", fields_eint, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_rho
        = storage.grad_rho.get().rankgather_computefield(sched);
    writer.write_field("grad_rho", grad_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dx_v = storage.dx_v.get().rankgather_computefield(sched);
    writer.write_field("dx_v", dx_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dy_v = storage.dy_v.get().rankgather_computefield(sched);
    writer.write_field("dy_v", dy_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dz_v = storage.dz_v.get().rankgather_computefield(sched);
    writer.write_field("dz_v", dz_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_P
        = storage.grad_P.get().rankgather_computefield(sched);
    writer.write_field("grad_P", grad_P, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrho = storage.dtrho.get().rankgather_computefield(sched);
    writer.write_field("dtrho", dtrho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dtrhov
        = storage.dtrhov.get().rankgather_computefield(sched);
    writer.write_field("dtrhov", dtrhov, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrhoe
        = storage.dtrhoe.get().rankgather_computefield(sched);
    writer.write_field("dtrhoe", dtrhoe, num_obj * block_size);
}

template class shammodels::basegodunov::Solver<f64_3, i64_3>;
