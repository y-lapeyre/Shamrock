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
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_L2_dist() -> field_val {

    auto get_gamma = [&]() -> Tscal {
        using SolverConfigEOS     = typename Config::EOSConfig;
        using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
        if (SolverEOS_Adiabatic *eos_config
            = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {
            return eos_config->gamma;
        }
        shambase::throw_with_loc<std::invalid_argument>(
            "The sod analysis is only available for adiabatic EOS");
        return {};
    };

    Tscal gamma = get_gamma();

    auto rho_h = [&](Tscal h) {
        return shamrock::sph::rho_h(solver_config.gpart_mass, h, Kernel::hfactd);
    };

    auto &sched = scheduler();

    const u32 ixyz   = sched.pdl.template get_field_idx<Tvec>("xyz");
    const u32 ihpart = sched.pdl.template get_field_idx<Tscal>("hpart");
    const u32 ivxyz  = sched.pdl.template get_field_idx<Tvec>("vxyz");
    const u32 iuint  = sched.pdl.template get_field_idx<Tscal>("uint");

    Tscal sum_L2_rho = 0;
    Tvec sum_L2_v    = {0, 0, 0};
    Tscal sum_L2_P   = 0;
    Tscal N          = 0;

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            auto &xyz_buf   = pdat.get_field_buf_ref<Tvec>(ixyz);
            auto &hpart_buf = pdat.get_field_buf_ref<Tscal>(ihpart);
            auto &vxyz_buf  = pdat.get_field_buf_ref<Tvec>(ivxyz);
            auto &uint_buf  = pdat.get_field_buf_ref<Tscal>(iuint);

            u32 len = pdat.get_obj_cnt();

            {
                auto acc_xyz   = xyz_buf.copy_to_stdvec();
                auto acc_hpart = hpart_buf.copy_to_stdvec();
                auto acc_vxyz  = vxyz_buf.copy_to_stdvec();
                auto acc_uint  = uint_buf.copy_to_stdvec();

                for (u32 i = 0; i < len; i++) {

                    Tvec xyz  = acc_xyz[i];
                    Tscal h   = acc_hpart[i];
                    Tvec vxyz = acc_vxyz[i];
                    Tscal u   = acc_uint[i];

                    Tscal rho = rho_h(h);
                    Tscal P   = shamphys::EOS_Adiabatic<Tscal>::pressure(gamma, rho, u);

                    Tscal x = sham::dot(xyz, direction) - x_ref;

                    if ((x + x_ref) > x_min && (x + x_ref) < x_max) {

                        auto result_sod = solution.get_value(time_val, x);

                        Tscal d_rho = rho - result_sod.rho;
                        Tscal d_P   = P - result_sod.P;
                        Tvec d_vxyz = vxyz - result_sod.vx * direction;

                        sum_L2_rho += d_rho * d_rho;
                        sum_L2_P += d_P * d_P;
                        sum_L2_v += d_vxyz * d_vxyz;
                        N += 1;
                    }
                }
            }
        });

    Tscal tot_N = shamalgs::collective::allreduce_sum(N);

    if (tot_N == 0) {
        shambase::throw_with_loc<std::runtime_error>("no particle in wanted region");
    }

    Tscal mpi_sum_L2_P   = shamalgs::collective::allreduce_sum(sum_L2_P) / tot_N;
    Tscal mpi_sum_L2_rho = shamalgs::collective::allreduce_sum(sum_L2_rho) / tot_N;
    Tvec mpi_sum_L2_v    = shamalgs::collective::allreduce_sum(sum_L2_v) / tot_N;

    return field_val{mpi_sum_L2_rho, mpi_sum_L2_v, mpi_sum_L2_P};
}

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

    return analysis_stage1{buff_tilt, buff_twist, buff_psi};
}

using namespace shammath;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M4>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M6>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M8>;
