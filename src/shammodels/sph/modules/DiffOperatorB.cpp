// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperatorB.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/DiffOperatorB.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperatorsB<Tvec, SPHKernel>::update_divB() {

    StackEntry stack_loc{};
    logger::debug_ln("SPH", "Updating divB");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 iB_on_rho_interf                               = ghost_layout.get_field_idx<Tvec>("B/rho");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idivB = pdl.get_field_idx<Tscal>("divB");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz
            = shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_B_on_rho   = mpdat.get_field_buf_ref<Tvec>(iB_on_rho_interf);
        sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_divB  = pdat.get_field_buf_ref<Tscal>(idivB);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute divB"};
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(pcache, cgh);

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor B_on_rho{buf_B_on_rho, cgh, sycl::read_only};
                sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                sycl::accessor omega{buf_omega, cgh, sycl::read_only};
                sycl::accessor divB{buf_divB, cgh, sycl::write_only, sycl::no_init};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute divB", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec B_on_rho_a    = B_on_rho[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    Tvec B_a    = B_on_rho_a * rho_a;
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tscal sum_nabla_B = 0;

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);

                        Tvec B_on_rho_b = B_on_rho[id_b];
                        Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                        Tvec B_b    = B_on_rho_b * rho_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;
                        Tvec B_ab        = (B_a - B_b);
                        sum_nabla_B +=  pmass * sycl::dot(B_ab, r_ab_unit * dWab_a);
                    });

                    Tscal sub_fact_a = rho_a * omega_a;
                   
                    divB[id_a] = -(1. / sub_fact_a) * sum_nabla_B;
                });
            });
        }
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperatorsB<Tvec, SPHKernel>::update_curlB() {

    StackEntry stack_loc{};
    logger::debug_ln("SPH", "Updating curlB");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 iB_on_rho_interf                               = ghost_layout.get_field_idx<Tvec>("B/rho");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 icurlB = pdl.get_field_idx<Tvec>("curlB");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz
            = shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_B_on_rho   = mpdat.get_field_buf_ref<Tvec>(iB_on_rho_interf);
        sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tvec> &buf_curlB  = pdat.get_field_buf_ref<Tvec>(icurlB);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute curlB"};
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(pcache, cgh);

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor B_on_rho{buf_B_on_rho, cgh, sycl::read_only};
                sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                sycl::accessor omega{buf_omega, cgh, sycl::read_only};
                sycl::accessor curlB{buf_curlB, cgh, sycl::write_only, sycl::no_init};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute curlB", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec B_on_rho_a    = B_on_rho[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    Tvec B_a    = B_on_rho_a * rho_a;
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tvec sum_nabla_cross_B{};

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec B_on_rho_b = B_on_rho[id_b];
                        Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                        Tvec B_b    = B_on_rho_b * rho_b;
                        Tvec B_ab   = B_a - B_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        sum_nabla_cross_B += pmass * sycl::cross(B_ab, dWab_a);
                    });

                    curlB[id_a] = -inv_rho_omega_a * sum_nabla_cross_B;
                });
            });
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperatorsB<f64_3, M4>;
template class shammodels::sph::modules::DiffOperatorsB<f64_3, M6>;
template class shammodels::sph::modules::DiffOperatorsB<f64_3, M8>;
