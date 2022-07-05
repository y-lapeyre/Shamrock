#pragma once

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

        for (u32 i = 0; i < f.size(); i++) {
            vec r = xyz.usm_data()[i];

            if (BBAA::is_particle_in_patch(r, std::get<0>(box), std::get<1>(box))) {
                f.usm_data()[i] = val;
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

        for (u32 i = 0; i < xyz.size(); i++) {
            vec r              = xyz.usm_data()[i];
            flt rkphi          = sycl::dot(r, k) + phase;
            vxyz.usm_data()[i] = ampl * sycl::sin(rkphi);
        }
    }
}

} // namespace generic::setup::modifiers