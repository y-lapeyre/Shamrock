// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierApplyDiscWarp.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyDiscWarp.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchData
shammodels::sph::modules::ModifierApplyDiscWarp<Tvec, SPHKernel>::next_n(u32 nmax) {

    using Config = SolverConfig<Tvec, SPHKernel>;
    Config solver_config;
    ShamrockCtx &ctx               = context;
    PatchScheduler &sched          = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchData tmp = parent->next_n(nmax);

    ////////////////////////// constants //////////////////////////
    constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;
    Tscal Rwarp          = Rwarp_;
    Tscal Hwarp          = Hwarp_;
    Tscal inclination    = inclination_;
    Tscal posangle       = posangle_;

    ////////////////////////// load data //////////////////////////
    sham::DeviceBuffer<Tvec> &buf_xyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_xyz  = buf_xyz.get_write_access(depends_list);
    auto acc_vxyz = buf_vxyz.get_write_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parralel_for(cgh, tmp.get_obj_cnt(), "Warp", [=](i32 id_a) {
            Tvec &xyz_a  = acc_xyz[id_a];
            Tvec &vxyz_a = acc_vxyz[id_a];

            Tscal r = sycl::sqrt(sycl::dot(xyz_a, xyz_a));

            Tvec k    = Tvec(-sycl::sin(posangle), sycl::cos(posangle), 0.);
            Tscal psi = 0.;

            // convert to radians (sycl functions take radians)
            Tscal incl_rad = inclination * shambase::constants::pi<Tscal> / 180.;
            Tscal R1       = 3.5;
            Tscal R2       = 6.5;
            Tscal R0       = 5.;
            Tscal A        = 0.5;
            Tscal lx, ly, lz;
            Tscal effective_inc;

            ly = 0.;
            if (r < R1) {
                effective_inc = 0.;
            } else if (r < R2 && r > R1) {
                effective_inc
                    = 0.5 * A
                      * (1. + sycl::sin(shambase::constants::pi<Tscal> * (r - R0) / (R2 - R1)));
            } else {
                lx            = A;
                effective_inc = incl_rad;
            }
            lz = sycl::sqrt(1. - lx * lx);

            Tvec w  = sycl::cross(k, xyz_a);
            Tvec wv = sycl::cross(k, vxyz_a);
            // Rodrigues' rotation formula
            xyz_a = xyz_a * sycl::cos(effective_inc) + w * sycl::sin(effective_inc)
                    + k * sycl::dot(k, xyz_a) * (1. - sycl::cos(effective_inc));
            vxyz_a = vxyz_a * sycl::cos(effective_inc) + wv * sycl::sin(effective_inc)
                     + k * sycl::dot(k, vxyz_a) * (1. - sycl::cos(effective_inc));
        });
    });

    buf_xyz.complete_event_state(e);
    buf_vxyz.complete_event_state(e);

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M4>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M6>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M8>;
