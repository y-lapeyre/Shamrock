// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "CL/sycl/accessor.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/utils/geometry_utils.hpp"
#include <stdexcept>
#include <tuple>
#include <vector>

namespace generic::setup::modifiers {

template <class T, class vec>
inline void set_value_in_box(PatchScheduler &sched, T val, std::string name, std::tuple<vec, vec> box) {

    for (auto &[pid, pdat] : sched.patch_data.owned_data) {

        PatchDataField<vec> &xyz = pdat.template get_field<vec>(sched.pdl.get_field_idx<vec>("xyz"));

        PatchDataField<T> &f = pdat.template get_field<T>(sched.pdl.get_field_idx<T>(name));


        {
            auto buf = f.get_sub_buf();
            sycl::host_accessor acc {*buf};

            auto buf_xyz = xyz.get_sub_buf();
            sycl::host_accessor acc_xyz {*buf_xyz};

            for (u32 i = 0; i < f.size(); i++) {
                vec r = acc_xyz[i];

                if (BBAA::is_particle_in_patch(r, std::get<0>(box), std::get<1>(box))) {
                    acc[i] = val;
                }
            }

        }
        
    }
}

template <class flt>
inline void pertub_eigenmode_wave(PatchScheduler &sched, std::tuple<flt, flt> ampls, sycl::vec<flt, 3> k, flt phase) {

    using vec = sycl::vec<flt, 3>;

    if (std::get<0>(ampls) != 0) {
        throw std::runtime_error("density perturbation not implemented");
    }

    for (auto &[pid, pdat] : sched.patch_data.owned_data) {

        PatchDataField<vec> &xyz  = pdat.template get_field<vec>(sched.pdl.get_field_idx<vec>("xyz"));
        PatchDataField<vec> &vxyz = pdat.template get_field<vec>(sched.pdl.get_field_idx<vec>("vxyz"));

        flt ampl = std::get<1>(ampls);

        {

            auto buf_xyz = xyz.get_sub_buf();
            sycl::host_accessor acc_xyz {*buf_xyz};

            auto buf_vxyz = vxyz.get_sub_buf();
            sycl::host_accessor acc_vxyz {*buf_vxyz};

            for (u32 i = 0; i < xyz.size(); i++) {
                vec r              = acc_xyz[i];
                flt rkphi          = sycl::dot(r, k) + phase;
                acc_vxyz[i] = ampl * sycl::sin(rkphi);
            }

        }

        
    }
}

} // namespace generic::setup::modifiers