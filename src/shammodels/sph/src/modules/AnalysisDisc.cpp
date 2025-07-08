// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisDisc.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage1(
    analysis_basis &basis) -> analysis_stage1 {

    sham::DeviceBuffer<Tscal> buff_tilt(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_twist(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_psi(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_tilt  = buff_tilt.get_write_access(depends_list); // @@@ necessary on device ?
    auto acc_twist = buff_twist.get_write_access(depends_list);
    auto acc_psi   = buff_psi.get_write_access(depends_list);

    auto acc_radius = basis.radius.copy_to_stdvec();
    auto acc_lx     = basis.lx.copy_to_stdvec();
    auto acc_ly     = basis.ly.copy_to_stdvec();
    auto acc_lz     = basis.lz.copy_to_stdvec();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        for (u32 i = 0; i < Nbin; i++) {
            Tscal radius = acc_radius[i];
            Tscal lx     = acc_lx[i];
            Tscal ly     = acc_ly[i];
            Tscal lz     = acc_lz[i];

            Tscal &tilt  = acc_tilt[i];
            Tscal &twist = acc_twist[i];
            Tscal &psi   = acc_psi[i];

            if (sycl::fabs(lz) > 0. && i < Nbin - 1) {
                tilt  = radius / lz; // @@@ missing a term
                twist = shambase::constants::pi<Tscal> * 0.5 * sycl::atan(lx / lz);
            }

            if (i > 0 && i < Nbin - 1) {
                Tscal radius_diff = acc_radius[i + 1] - acc_radius[i - 1];
                Tscal psi_x       = acc_lx[i + 1] - acc_lx[i - 1] / radius_diff;
                Tscal psi_y       = acc_ly[i + 1] - acc_ly[i - 1] / radius_diff;
                Tscal psi_z       = acc_lz[i + 1] - acc_lz[i - 1] / radius_diff;
                psi               = sycl::sqrt(psi_x * psi_x + psi_y * psi_y + psi_z * psi_z);
            }
        }
    });

    buff_tilt.complete_event_state(e);
    buff_twist.complete_event_state(e);
    buff_psi.complete_event_state(e);

    return analysis_stage1{std::move(buff_tilt), std::move(buff_twist), std::move(buff_psi)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage2(
    analysis_stage1 &stage1) -> analysis_stage2 {}

using namespace shammath;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M4>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M6>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M8>;
