// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExternalForces.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ExternalForces.hpp"

#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamunits/Constants.hpp"
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <hipSYCL/sycl/libkernel/builtins.hpp>

template<class Tvec, template<class> class SPHKernel>
using Module = shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>;

template<class Tvec, template<class> class SPHKernel>
void Module<Tvec, SPHKernel>::compute_ext_forces_indep_v(Tscal gpart_mass) {

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    modules::SinkParticlesUpdate<Tvec, SPHKernel> sink_update(context, solver_config, storage);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(iaxyz_ext);
        field.field_raz();
    });

    sink_update.compute_sph_forces(gpart_mass);

    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_LenseThrirring    = typename SolverConfigExtForce::LenseThirring;
    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                sycl::buffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sycl::buffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                    sycl::accessor axyz_ext{buf_axyz_ext, cgh, sycl::read_write};

                    Tscal mGM = -cmass * G;

                    shambase::parralel_for(
                        cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                            Tvec r_a       = xyz[gid];
                            Tscal abs_ra   = sycl::length(r_a);
                            Tscal abs_ra_3 = abs_ra * abs_ra * abs_ra;
                            axyz_ext[gid] += mGM * r_a / abs_ra_3;
                        });
                });
            });

        } else if (EF_LenseThrirring *ext_force = std::get_if<EF_LenseThrirring>(&var_force)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                sycl::buffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sycl::buffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                    sycl::accessor axyz_ext{buf_axyz_ext, cgh, sycl::read_write};

                    Tscal mGM = -cmass * G;

                    shambase::parralel_for(
                        cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                            Tvec r_a       = xyz[gid];
                            Tscal abs_ra   = sycl::length(r_a);
                            Tscal abs_ra_3 = abs_ra * abs_ra * abs_ra;
                            axyz_ext[gid] += mGM * r_a / abs_ra_3;
                        });
                });
            });
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void Module<Tvec, SPHKernel>::add_ext_forces(Tscal gpart_mass) {
    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::buffer<Tvec> &buf_axyz     = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sycl::buffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor axyz{buf_axyz, cgh, sycl::read_write};
            sycl::accessor axyz_ext{buf_axyz_ext, cgh, sycl::read_only};

            shambase::parralel_for(
                cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                    axyz[gid] += axyz_ext[gid];
                });
        });
    });


    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_LenseThrirring    = typename SolverConfigExtForce::LenseThirring;

    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_LenseThrirring *ext_force = std::get_if<EF_LenseThrirring>(&var_force)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();
            Tscal c     = solver_config.get_constant_c();
            Tscal GM = cmass*G;


            logger::raw_ln("S",ext_force->a_spin * GM*GM *ext_force->dir_spin/ (c*c*c));

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                sycl::buffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sycl::buffer<Tvec> &buf_vxyz      = pdat.get_field_buf_ref<Tvec>(ivxyz);
                sycl::buffer<Tvec> &buf_axyz     = pdat.get_field_buf_ref<Tvec>(iaxyz);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                    sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
                    sycl::accessor axyz{buf_axyz, cgh, sycl::read_write};

                    Tvec S = ext_force->a_spin * GM*GM *ext_force->dir_spin/ (c*c*c);

                    shambase::parralel_for(
                        cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                            Tvec r_a       = xyz[gid];
                            Tvec v_a       = vxyz[gid];
                            Tscal abs_ra   = sycl::length(r_a);
                            Tscal abs_ra_2 = abs_ra * abs_ra;
                            Tscal abs_ra_3 = abs_ra_2 * abs_ra;
                            Tscal abs_ra_5 = abs_ra_2 * abs_ra_2 * abs_ra;

                            Tvec omega_a = (2*S/abs_ra_3) - (6 * shambase::sycl_utils::g_sycl_dot(S, r_a)*r_a)/abs_ra_5;
                            Tvec acc_lt = sycl::cross(v_a, omega_a);
                            axyz[gid] +=  acc_lt;

                        });
                });
            });
        }
    }
}

using namespace shammath;
template class shammodels::sph::modules::ExternalForces<f64_3, M4>;
template class shammodels::sph::modules::ExternalForces<f64_3, M6>;