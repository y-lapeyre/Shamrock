// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "BasicGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
namespace shammodels::sph {

    void BasicGas::dump_vtk(std::string dump_name) {

        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        scheduler().for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<vec> pos(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(
                pos, get_check_ref(pdat.get_field<vec>(0).get_buf()), ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writer.write_points(pos, num_obj);
    }

    void BasicGas::apply_position_boundary(){

        shamrock::SchedulerUtility integrators(scheduler());

        const u32 ixyz       = scheduler().pdl.get_field_idx<vec>("xyz");
        auto [bmin, bmax] = scheduler().get_box_volume<vec>();
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});


    }

    void BasicGas::evolve(f64 dt) {

        using namespace shamrock::patch;

        const u32 ixyz       = scheduler().pdl.get_field_idx<vec>("xyz");
        const u32 ivxyz      = scheduler().pdl.get_field_idx<vec>("vxyz");
        const u32 iaxyz      = scheduler().pdl.get_field_idx<vec>("axyz");
        const u32 iaxyz_old  = scheduler().pdl.get_field_idx<vec>("axyz_old");
        const u32 iuint      = scheduler().pdl.get_field_idx<flt>("uint");
        const u32 iduint     = scheduler().pdl.get_field_idx<flt>("duint");
        const u32 iduint_loc = scheduler().pdl.get_field_idx<flt>("duint_loc");
        const u32 ihpart     = scheduler().pdl.get_field_idx<flt>("hpart");

        shamrock::SchedulerUtility integrators(scheduler());

        // forward euler step f dt/2
        integrators.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        integrators.fields_forward_euler<vec>(iuint, iduint, dt / 2);

        // forward euler step positions dt
        integrators.fields_forward_euler<vec>(ixyz, ivxyz, dt);

        // forward euler step f dt/2
        integrators.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        integrators.fields_forward_euler<vec>(iuint, iduint, dt / 2);

        // swap der
        integrators.fields_swap<vec>(iaxyz, iaxyz_old);

        // periodic box (This one should be in a apply boundary function or something like this)
        // apply_position boundary ?
        apply_position_boundary();

        // update h

        // compute pressure

        // compute force

        // corrector
        integrators.fields_leapfrog_corrector<vec>(ivxyz, iaxyz, iaxyz_old, dt / 2);
        integrators.fields_leapfrog_corrector<flt>(iuint, iduint, iduint_loc, dt / 2);

        // if delta too big jump to compute force
    }

} // namespace shammodels::sph