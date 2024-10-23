// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConservativeCheck.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ConservativeCheck.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"
#include <hipSYCL/sycl/buffer.hpp>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ConservativeCheck<Tvec, SPHKernel>::check_conservation() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;
    using Sink = SinkParticle<Tvec>;

    bool has_B_field   = solver_config.has_field_B_on_rho();
    bool has_psi_field = solver_config.has_field_psi_on_ch();

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");
    const u32 iB_on_rho = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : 0;
    const u32 idB       = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
    const u32 ipsi      = (has_psi_field) ? pdl.get_field_idx<Tscal>("psi/ch") : 0;
    const u32 idpsi     = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;

    std::string cv_checks = "convervation infos :\n";

    ///////////////////////////////////
    // momentum check :
    ///////////////////////////////////
    Tvec tmpp{0, 0, 0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(ivxyz);
        tmpp += field.compute_sum();
    });
    Tvec sum_p = gpart_mass * shamalgs::collective::allreduce_sum(tmpp);

    if (shamcomm::world_rank() == 0) {
        if (!storage.sinks.is_empty()) {
            std::vector<Sink> &sink_parts = storage.sinks.get();
            for (Sink &s : sink_parts) {
                sum_p += s.mass * s.velocity;
            }
        }
        cv_checks += shambase::format("    sum v = {}\n", sum_p);
    }

    ///////////////////////////////////
    // force sum check :
    ///////////////////////////////////
    Tvec tmpa{0, 0, 0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(iaxyz);
        tmpa += field.compute_sum();
    });
    Tvec sum_a = gpart_mass * shamalgs::collective::allreduce_sum(tmpa);

    if (shamcomm::world_rank() == 0) {
        if (!storage.sinks.is_empty()) {
            std::vector<Sink> &sink_parts = storage.sinks.get();
            for (Sink &s : sink_parts) {
                sum_a += s.mass * (s.sph_acceleration + s.ext_acceleration);
            }
        }
        cv_checks += shambase::format("    sum a = {}\n", sum_a);
    }

    ///////////////////////////////////
    // energy check :
    ///////////////////////////////////
    Tscal tmpe{0};
    Tscal tmp_e      = 0;
    Tscal tmp_ekin   = 0;
    Tscal tmp_etherm = 0;
    Tscal tmp_emag   = 0;
    Tscal tmp_divB   = 0;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tscal> &field_u = pdat.get_field<Tscal>(iuint);
        PatchDataField<Tvec> &field_v  = pdat.get_field<Tvec>(ivxyz);
        PatchDataField<Tscal> &field_h = pdat.get_field<Tscal>(ihpart);
        PatchDataField<Tvec> &field_B  = pdat.get_field<Tvec>(iB_on_rho);

        Tscal mu_0        = solver_config.get_constant_mu_0();
        const Tscal m     = solver_config.gpart_mass;
        const Tscal hfact = Kernel::hfactd;
        Tscal pmass       = gpart_mass;

        // const u32 size_field_h = field_h.size();
        // Tvec vec_hfact = sycl::vec<Tscal>(size_field_h);
        // Tvec buff_hfact = shamalgs::memory::vec_to_buf(vec_hfact);
        // PatchDataField<Tscal> pdf_hfact = PatchDataField<Tscal>(buff_hfact);
        // PatchDataField<Tscal> &field_rho = m * (hfact / field_h) * (hfact / field_h) * (hfact /
        // field_h);
        sycl::buffer<Tscal> temp_e(pdat.get_obj_cnt());
        sycl::buffer<Tscal> temp_ekin(pdat.get_obj_cnt());
        sycl::buffer<Tscal> temp_etherm(pdat.get_obj_cnt());
        sycl::buffer<Tscal> temp_emag(pdat.get_obj_cnt());
        sycl::buffer<Tscal> temp_divB(pdat.get_obj_cnt());

        shamsys::instance::get_compute_queue().submit([&, pmass](sycl::handler &cgh) {
            sycl::accessor u{*field_u.get_buf(), cgh, sycl::read_only};
            sycl::accessor v{*field_v.get_buf(), cgh, sycl::read_only};
            sycl::accessor h{*field_u.get_buf(), cgh, sycl::read_only};
            sycl::accessor B_on_rho{*field_B.get_buf(), cgh, sycl::read_only};
            sycl::accessor e{temp_e, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor ekin{temp_ekin, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor etherm{temp_etherm, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor emag{temp_emag, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor divB{temp_divB, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                Tscal rho    = pmass * (hfact / h[item]) * (hfact / h[item]) * (hfact / h[item]);
                ekin[item]   = 0.5 * pmass * sycl::dot(v[item], v[item]);
                etherm[item] = pmass * u[item];
                emag[item]   = 0.5 * sycl::dot(B_on_rho[item], B_on_rho[item]) * rho / mu_0;
                e[item]      = pmass * u[item] + 0.5 * pmass * sycl::dot(v[item], v[item])
                          + 0.5 * sycl::dot(B_on_rho[item], B_on_rho[item]) * rho / mu_0;
                divB[item] = 0;
            });
        });
        Tscal e_pkin = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_ekin, 0, pdat.get_obj_cnt());
        Tscal e_ptherm = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_etherm, 0, pdat.get_obj_cnt());
        Tscal e_pmag = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_emag, 0, pdat.get_obj_cnt());
        Tscal e_p = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_e, 0, pdat.get_obj_cnt());
        tmp_ekin += e_pkin;
        tmp_etherm += e_ptherm;
        tmp_emag += e_pmag;
        tmp_e += e_p;
    });

    Tscal sum_ekin   = shamalgs::collective::allreduce_sum(tmp_ekin);
    Tscal sum_etherm = shamalgs::collective::allreduce_sum(tmp_etherm);
    Tscal sum_emag   = shamalgs::collective::allreduce_sum(tmp_emag);
    Tscal sum_e      = shamalgs::collective::allreduce_sum(tmp_e);

    if (shamcomm::world_rank() == 0) {
        cv_checks += shambase::format("    sum e = {}\n", sum_e);
        cv_checks += shambase::format("    sum ekin = {}\n", sum_ekin);
        cv_checks += shambase::format("    sum etherm = {}\n", sum_etherm);
        cv_checks += shambase::format("    sum emag = {}\n", sum_emag);
    }

    Tscal pmass  = gpart_mass;
    Tscal tmp_de = 0;
    scheduler().for_each_patchdata_nonempty([&, pmass](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tscal> &field_u  = pdat.get_field<Tscal>(iuint);
        PatchDataField<Tvec> &field_v   = pdat.get_field<Tvec>(ivxyz);
        PatchDataField<Tscal> &field_du = pdat.get_field<Tscal>(iduint);
        PatchDataField<Tvec> &field_a   = pdat.get_field<Tvec>(iaxyz);

        sycl::buffer<Tscal> temp_de(pdat.get_obj_cnt());

        shamsys::instance::get_compute_queue().submit([&, pmass](sycl::handler &cgh) {
            sycl::accessor u{*field_u.get_buf(), cgh, sycl::read_only};
            sycl::accessor du{*field_du.get_buf(), cgh, sycl::read_only};
            sycl::accessor v{*field_v.get_buf(), cgh, sycl::read_only};
            sycl::accessor a{*field_a.get_buf(), cgh, sycl::read_only};
            sycl::accessor de{temp_de, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                de[item] = pmass * (sycl::dot(v[item], a[item]) + du[item]);
            });
        });

        Tscal de_p = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_de, 0, pdat.get_obj_cnt());
        tmp_de += de_p;
    });

    Tscal de = shamalgs::collective::allreduce_sum(tmp_de);

    if (shamcomm::world_rank() == 0) {
        cv_checks += shambase::format("    sum de = {}", de);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sph::Model", cv_checks);
    }
}

using namespace shammath;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M4>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M6>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M8>;
