// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "BasicGas.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambase/memory.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
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

    u64 BasicGas::count_particles(){
        u64 part_cnt = 0;
        using namespace shamrock::patch;
        scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            part_cnt += pdat.get_obj_cnt();
        });
        return shamalgs::collective::allreduce_sum(part_cnt);
    }

    void BasicGas::apply_position_boundary(){

        shamrock::SchedulerUtility integrators(scheduler());

        const u32 ixyz       = scheduler().pdl.get_field_idx<vec>("xyz");
        auto [bmin, bmax] = scheduler().get_box_volume<vec>();
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});


    }

    void BasicGas::evolve(f64 dt) {

        using namespace shamrock;
        using namespace shamrock::patch;

        const u32 ixyz       = scheduler().pdl.get_field_idx<vec>("xyz");
        const u32 ivxyz      = scheduler().pdl.get_field_idx<vec>("vxyz");
        const u32 iaxyz      = scheduler().pdl.get_field_idx<vec>("axyz");
        const u32 iuint      = scheduler().pdl.get_field_idx<flt>("uint");
        const u32 iduint     = scheduler().pdl.get_field_idx<flt>("duint");
        const u32 ihpart     = scheduler().pdl.get_field_idx<flt>("hpart");

        shamrock::SchedulerUtility utility(scheduler());

        // forward euler step f dt/2
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);

        // forward euler step positions dt
        utility.fields_forward_euler<vec>(ixyz, ivxyz, dt);

        // forward euler step f dt/2
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);

        // save old acceleration
        ComputeField<vec> old_axyz = utility.save_field<vec>(iaxyz, "axyz_old");
        ComputeField<flt> old_duint = utility.save_field<flt>(iaxyz, "duint_old");

        
        apply_position_boundary();

        u64 Npart_all = count_particles(); 

        // update h

        // compute pressure

        // compute force
        scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor acc_f{
                    shambase::get_check_ref(pdat.get_field<vec>(iaxyz).get_buf())
                    , cgh, sycl::write_only};

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid     = (u32)item.get_id();
                    acc_f[item] = vec{1,1,1};
                });
            });
        });


        ComputeField<flt> vepsilon_v_sq = utility.make_compute_field<flt>("vmean epsilon_v^2", 1);
        ComputeField<flt> uepsilon_u_sq = utility.make_compute_field<flt>("umean epsilon_u^2", 1);

        // corrector
        utility.fields_leapfrog_corrector<vec>(ivxyz, iaxyz, old_axyz,vepsilon_v_sq, dt / 2);
        utility.fields_leapfrog_corrector<flt>(iuint, iduint, old_duint,uepsilon_u_sq, dt / 2);

        flt rank_veps_v = sycl::sqrt(vepsilon_v_sq.compute_rank_max());
        flt rank_ueps_u = sycl::sqrt(uepsilon_u_sq.compute_rank_max());

        ///////////////////////////////////////////
        // compute means //////////////////////////
        ///////////////////////////////////////////

        flt sum_vsq = utility.compute_rank_dot_sum<vec>(ivxyz);
        flt sum_usq = utility.compute_rank_dot_sum<flt>(iuint);

        flt vmean_sq = shamalgs::collective::allreduce_sum(sum_vsq)/flt(Npart_all);
        flt umean_sq = shamalgs::collective::allreduce_sum(sum_usq)/flt(Npart_all);

        flt rank_eps_v = 0;
        flt rank_eps_u = 0;
        if(vmean_sq > 0){
            rank_eps_v = rank_veps_v / sycl::sqrt(vmean_sq);
        }
        if(umean_sq > 0){
            rank_eps_v = rank_ueps_u / sycl::sqrt(umean_sq);
        }

        flt eps_v = shamalgs::collective::allreduce_max(rank_eps_v);
        flt eps_u = shamalgs::collective::allreduce_max(rank_eps_u);

        // if delta too big jump to compute force
    }

} // namespace shammodels::sph