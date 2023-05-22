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
#include "shammodels/BasicSPHGhosts.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"











namespace shammodels::sph {

    template<class vec>
    shamrock::LegacyVtkWritter start_dump(PatchScheduler &sched, std::string dump_name) {
        StackEntry stack_loc{};
        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = sched.get_rank_count();

        logger::debug_mpi_ln("sph::BasicGas", "rank count =",num_obj);

        std::unique_ptr<sycl::buffer<vec>> pos = sched.rankgather_field<vec>(0);

        writer.write_points(pos, num_obj);
        
        return writer;
    }

    void vtk_dump_add_patch_id(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
        StackEntry stack_loc{};
        
        u64 num_obj = sched.get_rank_count();

        using namespace shamrock::patch;

        if(num_obj > 0){
            // TODO aggregate field ?
            sycl::buffer<u64> idp(num_obj);

            u64 ptr = 0; // TODO accumulate_field() in scheduler ?
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                using namespace shamalgs::memory;
                using namespace shambase;

                write_with_offset_into(idp, cur_p.id_patch, ptr, pdat.get_obj_cnt());

                ptr += pdat.get_obj_cnt();
            });

            writter.write_field("patchid", idp, num_obj);
        }else{
            writter.write_field_no_buf<u64>("patchid");
        }
    }

    void vtk_dump_add_worldrank(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
        StackEntry stack_loc{};
        
        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        if(num_obj > 0){

            // TODO aggregate field ?
            sycl::buffer<u32> idp(num_obj);

            u64 ptr = 0; // TODO accumulate_field() in scheduler ?
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                using namespace shamalgs::memory;
                using namespace shambase;

                write_with_offset_into(idp, shamsys::instance::world_rank, ptr, pdat.get_obj_cnt());

                ptr += pdat.get_obj_cnt();
            });

            writter.write_field("world_rank", idp, num_obj);

        }else{
            writter.write_field_no_buf<u32>("world_rank");
        }
    }

    template<class T>
    void vtk_dump_add_compute_field(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter, shamrock::ComputeField<T> & field, std::string field_dump_name) {
        StackEntry stack_loc{};
        
        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        std::unique_ptr<sycl::buffer<T>> field_vals = field.rankgather_computefield(sched);

        writter.write_field(field_dump_name, field_vals, num_obj);
    }

    template<class T>
    void vtk_dump_add_field(PatchScheduler &sched,
                            shamrock::LegacyVtkWritter &writter,
                            u32 field_idx,
                            std::string field_dump_name) {
        StackEntry stack_loc{};
        
        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        std::unique_ptr<sycl::buffer<T>> field_vals = sched.rankgather_field<T>(field_idx);

        writter.write_field(field_dump_name, field_vals, num_obj);
    }

    u64 BasicGas::count_particles() {
        StackEntry stack_loc{};
        
        u64 part_cnt = scheduler().get_rank_count();
        return shamalgs::collective::allreduce_sum(part_cnt);
    }

    void BasicGas::apply_position_boundary(SerialPatchTree<vec> &sptree) {
        StackEntry stack_loc{};

        shamrock::SchedulerUtility integrators(scheduler());
        shamrock::ReattributeDataUtility reatrib(scheduler());

        const u32 ixyz    = scheduler().pdl.get_field_idx<vec>("xyz");
        auto [bmin, bmax] = scheduler().get_box_volume<vec>();
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});

        reatrib.reatribute_patch_objects(sptree, "xyz");
    }

    void BasicGas::evolve(f64 dt, DumpOption dump_opt) {

        logger::info_ln("sph::BasicGas", ">>> Step :", dt);

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;

        const u32 ixyz   = scheduler().pdl.get_field_idx<vec>("xyz");
        const u32 ivxyz  = scheduler().pdl.get_field_idx<vec>("vxyz");
        const u32 iaxyz  = scheduler().pdl.get_field_idx<vec>("axyz");
        const u32 iuint  = scheduler().pdl.get_field_idx<flt>("uint");
        const u32 iduint = scheduler().pdl.get_field_idx<flt>("duint");
        const u32 ihpart = scheduler().pdl.get_field_idx<flt>("hpart");

        shamrock::SchedulerUtility utility(scheduler());

        // forward euler step f dt/2
        logger::info_ln("sph::BasicGas", "forward euler step f dt/2");
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);

        // forward euler step positions dt
        logger::info_ln("sph::BasicGas", "forward euler step positions dt");
        utility.fields_forward_euler<vec>(ixyz, ivxyz, dt);

        // forward euler step f dt/2
        logger::info_ln("sph::BasicGas", "forward euler step f dt/2");
        utility.fields_forward_euler<vec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<flt>(iuint, iduint, dt / 2);

        SerialPatchTree<vec> sptree = SerialPatchTree<vec>::build(scheduler());
        sptree.attach_buf();

        logger::info_ln("sph::BasicGas", "apply_position_boundary()");
        apply_position_boundary(sptree);

        // save old acceleration
        logger::info_ln("sph::BasicGas", "save old fields");
        ComputeField<vec> old_axyz  = utility.save_field<vec>(iaxyz, "axyz_old");
        ComputeField<flt> old_duint = utility.save_field<flt>(iduint, "duint_old");

        u64 Npart_all = count_particles();


        shamrock::patch::PatchField<flt> interactR_patch =
            scheduler().map_owned_to_patch_field_simple<flt>(
                [&](const Patch p, PatchData &pdat) -> flt {
                    if (!pdat.is_empty()) {
                        return pdat.get_field<flt>(ihpart).compute_max()*htol_up_tol*Rkern;
                    }else{
                        return shambase::VectorProperties<flt>::get_min();
                    }
                });

        PatchtreeField<flt> interactR_mpi_tree = sptree.make_patch_tree_field(
            scheduler(),
            shamsys::instance::get_compute_queue(),
            interactR_patch,
            [](flt h0, flt h1, flt h2, flt h3, flt h4, flt h5, flt h6, flt h7) {
                return shambase::sycl_utils::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
            });

        BasicGasPeriodicGhostHandler<vec> interf_handle (scheduler());

        auto interf_build_cache = interf_handle.make_interface_cache(sptree, interactR_mpi_tree,interactR_patch);

        InterfacesUtility interf_utils(scheduler());

        auto interf_xyz = interf_handle.build_communicate_positions(interf_build_cache);
        auto merged_xyz = interf_handle.merge_position_buf(std::move(interf_xyz));

        constexpr u32 reduc_level = 5;

        using RTree = RadixTree<u_morton, vec, 3>;

        shambase::DistributedData<RTree> trees = 
        merged_xyz.map<RTree>([&](u64 id, shamrock::MergedPatchDataField<vec> & merged){

            vec bmin = merged.bounds->lower;
            vec bmax = merged.bounds->upper; 

            RTree tree(
                shamsys::instance::get_compute_queue(), 
                {bmin,bmax},
                merged.field.get_buf(),
                merged.field.get_obj_cnt(),
                reduc_level);

            return tree;
        });

        trees.for_each([&](u64 id,RTree & tree ){
            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());
        });

        ComputeField<flt> _epsilon_h = utility.make_compute_field<flt>("epsilon_h", 1,flt(100));
        ComputeField<flt> _h_old = utility.save_field<flt>(ihpart, "h_old");
        ComputeField<flt> omega = utility.make_compute_field<flt>("omega", 1);

        for(u32 iter_h = 0; iter_h < 5; iter_h ++){
            //iterate smoothing lenght
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
                logger::debug_ln("SPHLeapfrog","patch : n°",p.id_patch,"->","h iteration");

                sycl::buffer<flt> & eps_h = shambase::get_check_ref(_epsilon_h.get_buf(p.id_patch));
                sycl::buffer<flt> & hold = shambase::get_check_ref(_h_old.get_buf(p.id_patch));
                sycl::buffer<flt> & omega_h = shambase::get_check_ref(omega.get_buf(p.id_patch));

                sycl::buffer<flt> & hnew = shambase::get_check_ref(pdat.get_field<flt>(ihpart).get_buf());
                sycl::buffer<vec> & merged_r = shambase::get_check_ref(merged_xyz.get(p.id_patch).field.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree tree = trees.get(p.id_patch);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                    tree::ObjectIterator particle_looper(tree,cgh);

                    sycl::accessor eps {eps_h, cgh,sycl::read_write};
                    sycl::accessor r {merged_r, cgh,sycl::read_only};
                    sycl::accessor h_new {hnew, cgh,sycl::read_write};
                    sycl::accessor h_old {hold, cgh,sycl::read_only};

                    const flt part_mass = gpart_mass;
                    const flt h_max_tot_max_evol = htol_up_tol;
                    const flt h_max_evol_p = htol_up_iter;
                    const flt h_max_evol_m = 1/htol_up_iter;

                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32)item.get_id(0);


                        if(eps[id_a] > 1e-6){

                            vec xyz_a = r[id_a]; // could be recovered from lambda

                            flt h_a = h_new[id_a];

                            vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            flt rho_sum = 0;
                            flt sumdWdh = 0;

                            particle_looper.rtree_for([&](u32, vec bmin,vec bmax) -> bool {
                                return shammath::domain_are_connected(bmin,bmax,inter_box_a_min,inter_box_a_max);
                            },[&](u32 id_b){
                                flt rab = sycl::distance( xyz_a , r[id_b]);

                                if(rab > h_a*Kernel::Rkern) { 
                                    return;
                                }

                                rho_sum += part_mass*Kernel::W(rab,h_a);
                                sumdWdh += part_mass*Kernel::dhW(rab,h_a);
                            });

                            using namespace shamrock::sph;

                            flt rho_ha = rho_h(part_mass, h_a);
                            flt new_h = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                            if(new_h < h_a*h_max_evol_m) new_h = h_max_evol_m*h_a;
                            if(new_h > h_a*h_max_evol_p) new_h = h_max_evol_p*h_a;
                            
                            flt ha_0 = h_old[id_a];
                            
                            if (new_h < ha_0*h_max_tot_max_evol) {
                                h_new[id_a] = new_h;
                                eps[id_a] = sycl::fabs(new_h - h_a)/ha_0;
                            }else{
                                h_new[id_a] = ha_0*h_max_tot_max_evol;
                                eps[id_a] = -1;
                            }

                        }
                    });


                });
                
            });
        }

        _epsilon_h.reset();
        _h_old.reset();


        //communicate fields
        /*
        PatchDataLayout interf_layout;
        interf_layout.add_field<flt>("hpart", 1);
        interf_layout.add_field<flt>("omega", 1);
        interf_layout.add_field<flt>("uint", 1);
        interf_layout.add_field<vec>("vxyz", 1);



        auto interf_h = interf_handle.build_interf_field<flt>(interf_build_cache, ihpart, [](i32_3){return flt(0);});
        auto interf_omega = interf_handle.build_interf_compute_field<flt>(interf_build_cache, omega, [](i32_3){return flt(0);});
        */





        // compute pressure        
        ComputeField<flt> pressure = utility.make_compute_field<flt>("pressure",1);



        // compute force
        logger::info_ln("sph::BasicGas", "compute force");
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_f{
                    shambase::get_check_ref(pdat.get_field<vec>(iaxyz).get_buf()),
                    cgh,
                    sycl::write_only};

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid     = (u32)item.get_id();
                    acc_f[item] = vec{1, 1, 1};
                });
            });
        });

        ComputeField<flt> vepsilon_v_sq = utility.make_compute_field<flt>("vmean epsilon_v^2", 1);
        ComputeField<flt> uepsilon_u_sq = utility.make_compute_field<flt>("umean epsilon_u^2", 1);

        // corrector
        logger::info_ln("sph::BasicGas", "leapfrog corrector");
        utility.fields_leapfrog_corrector<vec>(ivxyz, iaxyz, old_axyz, vepsilon_v_sq, dt / 2);
        utility.fields_leapfrog_corrector<flt>(iuint, iduint, old_duint, uepsilon_u_sq, dt / 2);
        
        flt rank_veps_v = sycl::sqrt(vepsilon_v_sq.compute_rank_max());
        flt rank_ueps_u = sycl::sqrt(uepsilon_u_sq.compute_rank_max());

        ///////////////////////////////////////////
        // compute means //////////////////////////
        ///////////////////////////////////////////

        flt sum_vsq = utility.compute_rank_dot_sum<vec>(ivxyz);
        flt sum_usq = utility.compute_rank_dot_sum<flt>(iuint);

        flt vmean_sq = shamalgs::collective::allreduce_sum(sum_vsq) / flt(Npart_all);
        flt umean_sq = shamalgs::collective::allreduce_sum(sum_usq) / flt(Npart_all);

        flt rank_eps_v = 0;
        flt rank_eps_u = 0;
        if (vmean_sq > 0) {
            rank_eps_v = rank_veps_v / sycl::sqrt(vmean_sq);
        }
        if (umean_sq > 0) {
            rank_eps_v = rank_ueps_u / sycl::sqrt(umean_sq);
        }

        flt eps_v = shamalgs::collective::allreduce_max(rank_eps_v);
        flt eps_u = shamalgs::collective::allreduce_max(rank_eps_u);

        // if delta too big jump to compute force


        ComputeField<flt> density = utility.make_compute_field<flt>("rho",1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_h{
                    shambase::get_check_ref(pdat.get_field<flt>(ihpart).get_buf()),
                    cgh,
                    sycl::read_only};

                sycl::accessor acc_rho{
                    shambase::get_check_ref(density.get_buf(p.id_patch)),
                    cgh,
                    sycl::write_only, sycl::no_init};
                const flt part_mass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid     = (u32)item.get_id();
                    using namespace shamrock::sph;
                    flt rho_ha = rho_h(part_mass, acc_h[gid]);
                    acc_rho[gid] = rho_ha;
                });
            });
        });

        if (dump_opt.vtk_do_dump) {
            shamrock::LegacyVtkWritter writter =
                start_dump<vec>(scheduler(), dump_opt.vtk_dump_fname);
            writter.add_point_data_section();

            u32 fnum = 0;
            if (dump_opt.vtk_dump_patch_id) {
                fnum += 2;
            }
            fnum++;
            fnum++;
            fnum++;
            fnum++;

            writter.add_field_data_section(fnum);

            if (dump_opt.vtk_dump_patch_id) {
                vtk_dump_add_patch_id(scheduler(), writter);
                vtk_dump_add_worldrank(scheduler(), writter);
            }


            vtk_dump_add_field<flt>(scheduler(), writter, ihpart, "h");
            vtk_dump_add_field<vec>(scheduler(), writter, ivxyz, "v");
            vtk_dump_add_field<vec>(scheduler(), writter, iaxyz, "a");

            vtk_dump_add_compute_field(scheduler(), writter, density, "rho");
        }
    }

} // namespace shammodels::sph