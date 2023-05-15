// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "BasicGas.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shamrock/legacy/patch/utility/serialpatchtree.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::sph {

    template<class vec>
    shamrock::LegacyVtkWritter start_dump(PatchScheduler & sched, std::string dump_name){
        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        sched.for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<vec> pos(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(
                pos, get_check_ref(pdat.get_field<vec>(0).get_buf()), ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writer.write_points(pos, num_obj);

        return writer;
    }

    void vtk_dump_add_patch_id(PatchScheduler & sched, shamrock::LegacyVtkWritter & writter){
        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        using namespace shamrock::patch;
        sched.for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<u64> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(idp,id_patch,ptr,pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("patchid", idp, num_obj);
    }

    void vtk_dump_add_worldrank(PatchScheduler & sched, shamrock::LegacyVtkWritter & writter){
        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        using namespace shamrock::patch;
        sched.for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<u32> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(idp,shamsys::instance::world_rank,ptr,pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("world_rank", idp, num_obj);
    }

    template<class T> void vtk_dump_add_field(PatchScheduler & sched, shamrock::LegacyVtkWritter & writter, u32 field_idx, std::string field_dump_name){
        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        using namespace shamrock::patch;
        sched.for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<T> field_vals(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(
                field_vals, get_check_ref(pdat.get_field<T>(field_idx).get_buf()), ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field(field_dump_name, field_vals, num_obj);
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


    

    void BasicGas::evolve(f64 dt, DumpOption dump_opt) {

        logger::info_ln("sph::BasicGas",">>> Step :",dt);

        StackEntry stack_loc{};

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
        logger::info_ln("sph::BasicGas","forward euler step f dt/2");
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);


        // forward euler step positions dt
        logger::info_ln("sph::BasicGas","forward euler step positions dt");
        utility.fields_forward_euler<vec>(ixyz, ivxyz, dt);

        // forward euler step f dt/2
        logger::info_ln("sph::BasicGas","forward euler step f dt/2");
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);

        // save old acceleration
        logger::info_ln("sph::BasicGas","save old fields");
        ComputeField<vec> old_axyz = utility.save_field<vec>(iaxyz, "axyz_old");
        ComputeField<flt> old_duint = utility.save_field<flt>(iduint, "duint_old");

        
        logger::info_ln("sph::BasicGas","apply_position_boundary()");
        apply_position_boundary();

        u64 Npart_all = count_particles(); 

        
        


        SerialPatchTree<vec> sptree = SerialPatchTree<vec>::build(scheduler());
        sptree.attach_buf();

        shamrock::patch::PatchField<flt> h_max_patch = scheduler().map_owned_to_patch_field_simple<flt>(
            [&](const Patch p , PatchData& pdat) -> flt{
                return pdat.get_field<flt>(ihpart).compute_max();
            }
        );

        PatchtreeField<flt> h_max_mpi_tree = sptree.make_patch_tree_field(
            scheduler(), 
            shamsys::instance::get_compute_queue(), 
            h_max_patch, 
            [](flt h0, flt h1, flt h2, flt h3, flt h4, flt h5, flt h6, flt h7){
                return shambase::sycl_utils::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
            });



        











        // update h

        // compute pressure

        // compute force
        logger::info_ln("sph::BasicGas","compute force");
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
        logger::info_ln("sph::BasicGas","leapfrog corrector");
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

        if(dump_opt.vtk_do_dump){
            shamrock::LegacyVtkWritter writter = start_dump<vec>(scheduler(), dump_opt.vtk_dump_fname);
            writter.add_point_data_section();

            u32 fnum = 0;
            if (dump_opt.vtk_dump_patch_id) {fnum += 2;}
            fnum ++;
            fnum ++;

            writter.add_field_data_section(fnum);

            if(dump_opt.vtk_dump_patch_id){
                vtk_dump_add_patch_id(scheduler(), writter);
                vtk_dump_add_worldrank(scheduler(), writter);
            }

            vtk_dump_add_field<vec>(scheduler(), writter, ivxyz, "v");
            vtk_dump_add_field<vec>(scheduler(), writter, iaxyz, "a");
        }

        
    }

} // namespace shammodels::sph