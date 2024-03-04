// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeGradient.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ComputeGradient.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

#include <utility>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraphLinkiterator;

namespace {

    template<class T, class Tvec, class TgridVec, class ACCField>
    inline std::array<T, 3> get_3d_grad(
        const u32 cell_global_id,
        const TgridVec delta_cell,
        const shambase::VecComponent<Tvec> dxfact,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &field_access) {

        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access[id_b];
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        T rho_i  = field_access[cell_global_id];
        T rho_xp = get_avg_neigh(graph_iter_xp);
        T rho_xm = get_avg_neigh(graph_iter_xm);
        T rho_yp = get_avg_neigh(graph_iter_yp);
        T rho_ym = get_avg_neigh(graph_iter_ym);
        T rho_zp = get_avg_neigh(graph_iter_zp);
        T rho_zm = get_avg_neigh(graph_iter_zm);

        T delta_rho_x_p = rho_xp - rho_i;
        T delta_rho_y_p = rho_yp - rho_i;
        T delta_rho_z_p = rho_zp - rho_i;

        T delta_rho_x_m = rho_i - rho_xm;
        T delta_rho_y_m = rho_i - rho_ym;
        T delta_rho_z_m = rho_i - rho_zm;

        T delta_rho_x_t = rho_xp - rho_xm;
        T delta_rho_y_t = rho_yp - rho_ym;
        T delta_rho_z_t = rho_zp - rho_zm;

        auto vanleer = [](T f) {
            return 4. * f * (1. - f);
        };

        T fact = 1. / (2 * dxfact * T(delta_cell.x()));

        T lim_slope_rho_x = vanleer(delta_rho_x_m / delta_rho_x_t) * delta_rho_x_t * fact;
        T lim_slope_rho_y = vanleer(delta_rho_y_m / delta_rho_y_t) * delta_rho_y_t * fact;
        T lim_slope_rho_z = vanleer(delta_rho_z_m / delta_rho_z_t) * delta_rho_z_t * fact;

        return {lim_slope_rho_x, lim_slope_rho_y, lim_slope_rho_z};
    }
} // namespace

template<class Tvec, class TgridVec>
void
shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::compute_grad_rho_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> result = utility.make_compute_field<Tvec>("gradient rho", AMRBlock::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);

        AMRGraph &graph_neigh_xp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        q.submit([&](sycl::handler &cgh) {
            AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
            AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
            AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
            AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
            AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
            AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

            sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
            sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};

            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor grad_rho{
                shambase::get_check_ref(result.get_buf(id)), cgh, sycl::write_only, sycl::no_init};

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            shambase::parralel_for(cgh, cell_count, "compute_grad_rho", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const TgridVec cblock_min = acc_block_min[block_id];
                const TgridVec cblock_max = acc_block_max[block_id];
                const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

                auto result = get_3d_grad<Tscal, Tvec>(
                    cell_global_id,
                    delta_cell,
                    dxfact,
                    graph_iter_xp,
                    graph_iter_xm,
                    graph_iter_yp,
                    graph_iter_ym,
                    graph_iter_zp,
                    graph_iter_zm,
                    rho);

                grad_rho[cell_global_id] = {result[0], result[1], result[2]};
            });
        });
    });

    storage.grad_rho.set(std::move(result));
}

template<class Tvec, class TgridVec>
void
shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::compute_grad_rhov_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> resultx = utility.make_compute_field<Tvec>("gradient dx rhov", AMRBlock::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });shamrock::ComputeField<Tvec> resulty = utility.make_compute_field<Tvec>("gradient dy rhov", AMRBlock::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });shamrock::ComputeField<Tvec> resultz = utility.make_compute_field<Tvec>("gradient dz rhov", AMRBlock::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irhov_ghost                                 = ghost_layout.get_field_idx<Tvec>("rhovel");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tvec> &buf_rhov = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_ghost);

        AMRGraph &graph_neigh_xp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        q.submit([&](sycl::handler &cgh) {
            AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
            AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
            AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
            AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
            AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
            AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

            sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
            sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};

            sycl::accessor rhovel{buf_rhov, cgh, sycl::read_only};
            sycl::accessor dx_rhovel{
                shambase::get_check_ref(resultx.get_buf(id)), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor dy_rhovel{
                shambase::get_check_ref(resulty.get_buf(id)), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor dz_rhovel{
                shambase::get_check_ref(resultz.get_buf(id)), cgh, sycl::write_only, sycl::no_init};

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            shambase::parralel_for(cgh, cell_count, "compute_grad_rho", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const TgridVec cblock_min = acc_block_min[block_id];
                const TgridVec cblock_max = acc_block_max[block_id];
                const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

                auto result = get_3d_grad<Tvec, Tvec>(
                    cell_global_id,
                    delta_cell,
                    dxfact,
                    graph_iter_xp,
                    graph_iter_xm,
                    graph_iter_yp,
                    graph_iter_ym,
                    graph_iter_zp,
                    graph_iter_zm,
                    rhovel);

                dx_rhovel[cell_global_id] = result[0];
                dy_rhovel[cell_global_id] = result[1];
                dz_rhovel[cell_global_id] = result[2];

            });
        });
    });

    storage.dx_rhov.set(std::move(resultx));
    storage.dy_rhov.set(std::move(resulty));
    storage.dz_rhov.set(std::move(resultz));
}

template<class Tvec, class TgridVec>
void
shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::compute_grad_rhoe_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> result = utility.make_compute_field<Tvec>("gradient rho rhoetot", AMRBlock::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irhoe_ghost                                 = ghost_layout.get_field_idx<Tscal>("rhoetot");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &buf_rhoe = mpdat.pdat.get_field_buf_ref<Tscal>(irhoe_ghost);

        AMRGraph &graph_neigh_xp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        q.submit([&](sycl::handler &cgh) {
            AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
            AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
            AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
            AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
            AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
            AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

            sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
            sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};

            sycl::accessor rhoe{buf_rhoe, cgh, sycl::read_only};
            sycl::accessor grad_rhoe{
                shambase::get_check_ref(result.get_buf(id)), cgh, sycl::write_only, sycl::no_init};

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            shambase::parralel_for(cgh, cell_count, "compute_grad_rho", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const TgridVec cblock_min = acc_block_min[block_id];
                const TgridVec cblock_max = acc_block_max[block_id];
                const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

                auto result = get_3d_grad<Tscal, Tvec>(
                    cell_global_id,
                    delta_cell,
                    dxfact,
                    graph_iter_xp,
                    graph_iter_xm,
                    graph_iter_yp,
                    graph_iter_ym,
                    graph_iter_zp,
                    graph_iter_zm,
                    rhoe);

                grad_rhoe[cell_global_id] = {result[0], result[1], result[2]};
            });
        });
    });

    storage.grad_rhoe.set(std::move(result));
}

template class shammodels::basegodunov::modules::ComputeGradient<f64_3, i64_3>;