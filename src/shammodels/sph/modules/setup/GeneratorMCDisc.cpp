// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GeneratorMCDisc.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shamalgs/random.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/GeneratorMCDisc.hpp"

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::GeneratorMCDisc<Tvec, SPHKernel>::DiscIterator::next()
    -> DiscOutput {

    Tscal fmax = f_func(r_out);

    auto find_r = [&]() {
        while (true) {
            Tscal u2 = shamalgs::random::mock_value<Tscal>(eng, 0, fmax);
            Tscal r  = shamalgs::random::mock_value<Tscal>(eng, r_in, r_out);
            if (u2 < f_func(r)) {
                return r;
            }
        }
    };

    auto theta = shamalgs::random::mock_value<Tscal>(eng, 0, _2pi);
    auto Gauss = shamalgs::random::mock_gaussian<Tscal>(eng);

    Tscal r = find_r();

    Tscal rot_speed = rot_profile(r);
    Tscal cs        = cs_profile(r);
    Tscal sigma     = sigma_profile(r);
    Tscal H         = H_profile(r);

    Tscal z = H * Gauss;

    auto pos = sycl::vec<Tscal, 3>{r * sycl::cos(theta), r * sycl::sin(theta), z};

    auto etheta = sycl::vec<Tscal, 3>{-pos.y(), pos.x(), 0};
    etheta /= sycl::length(etheta);

    auto vel = rot_speed * etheta;

    // Tscal fs  = 1. - sycl::sqrt(r_in / r);
    Tscal fs  = 1;
    Tscal rho = (sigma * fs) * sycl::exp(-z * z / (2 * H * H));

    DiscOutput out{pos, vel, cs, rho};

    // increase counter + check if finished
    current_index++;
    if (current_index == Npart) {
        done = true;
    }

    return out;
}

template<class Tvec, template<class> class SPHKernel>
bool shammodels::sph::modules::GeneratorMCDisc<Tvec, SPHKernel>::is_done() {
    return generator.is_done();
}

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchData
shammodels::sph::modules::GeneratorMCDisc<Tvec, SPHKernel>::next_n(u32 nmax) {

    using namespace shamrock::patch;
    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    auto has_pdat = [&]() {
        bool ret = false;
        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            ret = true;
        });
        return ret;
    };

    std::vector<DiscOutput> pos_data;

    // Fill pos_data if the scheduler has some patchdata in this rank
    if (!is_done()) {
        u64 loc_gen_count = (has_pdat()) ? nmax : 0;

        auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

        u64 skip_start = gen_info.head_offset;
        u64 gen_cnt    = loc_gen_count;
        u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

        logger::debug_ln(
            "GeneratorMCDisc",
            "generate : ",
            skip_start,
            gen_cnt,
            skip_end,
            "total",
            skip_start + gen_cnt + skip_end);

        generator.skip(skip_start);
        pos_data = generator.next_n(gen_cnt);
        generator.skip(skip_end);
    }

    // extract the pos from part_list
    std::vector<Tvec> vec_pos;
    std::vector<Tvec> vec_vel;
    std::vector<Tscal> vec_u;
    std::vector<Tscal> vec_h;
    std::vector<Tscal> vec_cs;

    for (DiscOutput o : pos_data) {
        vec_pos.push_back(o.pos);
        vec_vel.push_back(o.velocity);
        // vec_u.push_back(o.cs * o.cs / (/*solver.eos_gamma * */ (eos_gamma - 1)));
        Tscal h = shamrock::sph::h_rho(pmass, o.rho * 0.1, Kernel::hfactd);
        vec_h.push_back(h);
        vec_cs.push_back(o.cs);
    }

    // Make a patchdata from pos_data
    PatchData tmp(sched.pdl);
    if (!pos_data.empty()) {
        tmp.resize(pos_data.size());
        tmp.fields_raz();

        {
            u32 len                 = pos_data.size();
            PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
            sycl::buffer<Tvec> buf(vec_pos.data(), len);
            f.override(buf, len);
        }

        {
            u32 len                 = pos_data.size();
            PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
            sycl::buffer<Tvec> buf(vec_vel.data(), len);
            f.override(buf, len);
        }
        {
            u32 len = vec_pos.size();
            PatchDataField<Tscal> &f
                = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
            sycl::buffer<Tscal> buf(vec_h.data(), len);
            f.override(buf, len);
        }

        if (solver_config.is_eos_locally_isothermal()) {
            u32 len = vec_pos.size();
            PatchDataField<Tscal> &f
                = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("soundspeed"));
            sycl::buffer<Tscal> buf(vec_cs.data(), len);
            f.override(buf, len);
        }
    }

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::GeneratorMCDisc<f64_3, M4>;
template class shammodels::sph::modules::GeneratorMCDisc<f64_3, M6>;
template class shammodels::sph::modules::GeneratorMCDisc<f64_3, M8>;
