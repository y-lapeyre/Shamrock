// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "SPHModelSolver.hpp"
#include "shambase/time.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shammodels/BasicSPHGhosts.hpp"
#include "shammodels/SPHSolverImpl.hpp"
#include "shamrock/sph/SPHUtilities.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shammodels/BasicSPHGhosts.hpp"
#include "shamrock/tree/TreeTaversalCache.hpp"
#include "shamrock/sph/forces.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamrock/sph/sphpart.hpp"

template<class Tvec, template<class> class Kern>
using SPHSolve = shammodels::SPHModelSolver<Tvec, Kern>;



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

u64 count_particles(PatchScheduler & sched){
    StackEntry stack_loc{};
        
    u64 part_cnt = sched.get_rank_count();
    return shamalgs::collective::allreduce_sum(part_cnt);
};

template<class Tvec>
void apply_position_boundary(PatchScheduler & sched,SerialPatchTree<Tvec> &sptree) {
    StackEntry stack_loc{};

    shamrock::SchedulerUtility integrators(sched);
    shamrock::ReattributeDataUtility reatrib(sched);

    const u32 ixyz    = sched.pdl.get_field_idx<Tvec>("xyz");
    auto [bmin, bmax] = sched.get_box_volume<Tvec>();
    integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});

    reatrib.reatribute_patch_objects(sptree, "xyz");
}

template<class Tvec, template<class> class Kern>
auto SPHSolve<Tvec, Kern>::evolve_once(
    Tscal dt, bool enable_physics, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id)
    -> Tscal {
    //tmp_solver.set_cfl_cour(cfl_cour);
    //tmp_solver.set_cfl_force(cfl_force);
    //tmp_solver.set_particle_mass(gpart_mass);
    // tmp_solver.set_gamma(eos_gamma);

    struct DumpOption {
        bool vtk_do_dump;
        std::string vtk_dump_fname;
        bool vtk_dump_patch_id;
    };

    DumpOption dump_opt{do_dump, vtk_dump_name, vtk_dump_patch_id};

    logger::info_ln("sph::BasicGas", ">>> Step :", dt);

    shambase::Timer tstep;
    tstep.start();

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::SchedulerUtility utility(scheduler());

    // forward euler step f dt/2
    logger::info_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

    // forward euler step positions dt
    logger::info_ln("sph::BasicGas", "forward euler step positions dt");
    utility.fields_forward_euler<Tvec>(ixyz, ivxyz, dt);

    // forward euler step f dt/2
    logger::info_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

    SerialPatchTree<Tvec> sptree = SerialPatchTree<Tvec>::build(scheduler());
    sptree.attach_buf();

    logger::info_ln("sph::BasicGas", "apply_position_boundary()");
    apply_position_boundary(scheduler() ,sptree);


    u64 Npart_all =  count_particles(scheduler());

    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;
    SPHUtils sph_utils(scheduler());

    sph::BasicSPHGhostHandler<Tvec> interf_handle (scheduler());
    InterfacesUtility interf_utils(scheduler());
    
    auto interf_build_cache = sph_utils.build_interf_cache(interf_handle,sptree,htol_up_tol);
    auto merged_xyz = interf_handle.build_comm_merge_positions(interf_build_cache);

    constexpr u32 reduc_level = 3;

    using RTree = RadixTree<u_morton, Tvec>;

    SPHSolverImpl solver(context);

    shambase::DistributedData<RTree> trees = solver.make_merge_patch_trees(merged_xyz,reduc_level);

    ComputeField<Tscal> _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1,Tscal(100));
    ComputeField<Tscal> _h_old = utility.save_field<Tscal>(ihpart, "h_old");


        
    tree::ObjectCacheHandler hiter_caches{
        u64(10e9),
        [&](u64 patch_id){
            logger::debug_ln("SPHLeapfrog","patch : n°",patch_id,"->","gen cache");

            sycl::buffer<Tvec> & merged_r = shambase::get_check_ref(merged_xyz.get(patch_id).field.get_buf());
            sycl::buffer<Tscal> & hold = shambase::get_check_ref(_h_old.get_buf(patch_id));

            PatchData & pdat = scheduler().patch_data.get_pdat(patch_id);

            RTree & tree = trees.get(patch_id);

            tree::ObjectCache pcache = solver.build_hiter_neigh_cache(
                0, 
                pdat.get_obj_cnt(),
                merged_r, 
                hold, 
                tree,
                htol_up_tol);

            return pcache;
        }
    };


        for(u32 iter_h = 0; iter_h < 5; iter_h ++){
            NamedStackEntry stack_loc2 {"iterate smoothing lenght"};
            //iterate smoothing lenght
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
                logger::debug_ln("SPHLeapfrog","patch : n°",p.id_patch,"->","h iteration");

                sycl::buffer<Tscal> & eps_h = shambase::get_check_ref(_epsilon_h.get_buf(p.id_patch));
                sycl::buffer<Tscal> & hold = shambase::get_check_ref(_h_old.get_buf(p.id_patch));

                sycl::buffer<Tscal> & hnew = shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf());
                sycl::buffer<Tvec> & merged_r = shambase::get_check_ref(merged_xyz.get(p.id_patch).field.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree & tree = trees.get(p.id_patch);

                tree::ObjectCache & neigh_cache = hiter_caches.get_cache(p.id_patch);



                sph_utils.iterate_smoothing_lenght_cache(merged_r, hnew, hold, eps_h, range_npart, neigh_cache, gpart_mass, htol_up_tol, htol_up_iter);
                //sph_utils.iterate_smoothing_lenght_tree(merged_r, hnew, hold, eps_h, range_npart, tree, gpart_mass, htol_up_tol, htol_up_iter);
                
            });
        }

        //// compute omega
        ComputeField<Tscal> omega = utility.make_compute_field<Tscal>("omega", 1);
        {
            NamedStackEntry stack_loc2 {"compute omega"};

            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
                logger::debug_ln("SPHLeapfrog","patch : n°",p.id_patch,"->","h iteration");

                sycl::buffer<Tscal> & omega_h = shambase::get_check_ref(omega.get_buf(p.id_patch));

                sycl::buffer<Tscal> & hnew = shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf());
                sycl::buffer<Tvec> & merged_r = shambase::get_check_ref(merged_xyz.get(p.id_patch).field.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree & tree = trees.get(p.id_patch);

                tree::ObjectCache & neigh_cache = hiter_caches.get_cache(p.id_patch);

                sph_utils.compute_omega(merged_r, hnew, omega_h, range_npart, neigh_cache, gpart_mass);
                
            });
        }
        _epsilon_h.reset();
        _h_old.reset();
        hiter_caches.reset();












        PatchDataLayout interf_layout;
        interf_layout.add_field<Tscal>("hpart", 1);
        interf_layout.add_field<Tscal>("uint", 1);
        interf_layout.add_field<Tvec>("vxyz", 1);
        interf_layout.add_field<Tscal>("omega", 1);

        u32 ihpart_interf = 0;
        u32 iuint_interf  = 1;
        u32 ivxyz_interf  = 2;
        u32 iomega_interf = 3;

        

        using RTreeField = RadixTreeField<Tscal>;
        shambase::DistributedData<RTreeField> rtree_field_h;

        shambase::DistributedData<MergedPatchData> mpdat;

        tree::ObjectCacheHandler neigh_caches(u64(10e9),[&](u64 patch_id){

            logger::debug_ln("BasicSPH", "build particle cache id =",patch_id);

            NamedStackEntry cache_build_stack_loc{"build cache"};

            MergedPatchData & merged_patch = mpdat.get(patch_id);

            PatchData & pdat = scheduler().patch_data.get_pdat(patch_id);

            sycl::buffer<Tvec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(patch_id).field.get_buf());
            sycl::buffer<Tscal> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(ihpart_interf).get_buf());
            sycl::buffer<Tscal> & tree_field_hmax = shambase::get_check_ref(rtree_field_h.get(patch_id).radix_tree_field_buf);

            sycl::range range_npart{pdat.get_obj_cnt()};

            RTree & tree = trees.get(patch_id);

            tree::ObjectCache pcache = solver.build_neigh_cache(
                0, 
                pdat.get_obj_cnt(),
                buf_xyz, 
                buf_hpart, 
                tree, 
                tree_field_hmax);

            return pcache;

        });

        Tscal next_cfl = 0;

        
        u32 corrector_iter_cnt = 0;
        bool need_rerun_corrector = false;
        do{

            if(corrector_iter_cnt == 50){
                throw shambase::throw_with_loc<std::runtime_error>("the corrector has made over 50 loops, either their is a bug, either you are using a dt that is too large");
            }

                

            //communicate fields
            

            using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

            auto pdat_interf = interf_handle.template build_interface_native<PatchData>(interf_build_cache, 
                [&](u64 sender,u64 /*receiver*/,InterfaceBuildInfos binfo, sycl::buffer<u32> & buf_idx, u32 cnt){

                    PatchData pdat(interf_layout);

                    PatchData & sender_patch = scheduler().patch_data.get_pdat(sender);
                    PatchDataField<Tscal> & sender_omega = omega.get_field(sender);

                    sender_patch.get_field<Tscal>(ihpart).append_subset_to(buf_idx,cnt,pdat.get_field<Tscal>(ihpart_interf));
                    sender_patch.get_field<Tscal>(iuint) .append_subset_to(buf_idx,cnt,pdat.get_field<Tscal>(iuint_interf));
                    sender_patch.get_field<Tvec>(ivxyz) .append_subset_to(buf_idx,cnt,pdat.get_field<Tvec>(ivxyz_interf));
                    sender_omega.append_subset_to(buf_idx,cnt,pdat.get_field<Tscal>(iomega_interf));
                    
                    pdat.check_field_obj_cnt_match();

                    return pdat;
                }
            );

            shambase::DistributedDataShared<PatchData> interf_pdat = 
                interf_handle.communicate_pdat(interf_layout, std::move(pdat_interf));


            mpdat = interf_handle.template merge_native<PatchData,MergedPatchData>(
                std::move(interf_pdat), 
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchData & pdat){
                    
                    PatchData pdat_new(interf_layout);

                    u32 or_elem        = pdat.get_obj_cnt();
                    u32 total_elements = or_elem;

                    PatchDataField<Tscal> & cur_omega = omega.get_field(p.id_patch);

                    pdat_new.get_field<Tscal>(ihpart_interf).insert(pdat.get_field<Tscal>(ihpart));
                    pdat_new.get_field<Tscal>(iuint_interf ).insert(pdat.get_field<Tscal>(iuint));
                    pdat_new.get_field<Tvec>(ivxyz_interf ).insert(pdat.get_field<Tvec>(ivxyz));
                    pdat_new.get_field<Tscal>(iomega_interf).insert(cur_omega);
                    
                    pdat_new.check_field_obj_cnt_match();

                    return MergedPatchData{
                        or_elem,
                        total_elements,
                        std::move(pdat_new),
                        interf_layout
                    };

                }, [](MergedPatchData &mpdat, PatchData &pdat_interf){
                    mpdat.total_elements += pdat_interf.get_obj_cnt();
                    mpdat.pdat.insert_elements(pdat_interf);
                });



            // compute pressure        
            ComputeField<Tscal> pressure = utility.make_compute_field<Tscal>("pressure",1, [&](u64 id){
                return mpdat.get(id).total_elements;
            });

            {
                NamedStackEntry stack_loc{"compute eos"};
                
                mpdat.for_each([&](u64 id, MergedPatchData & mpdat){

                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor P {pressure.get_buf_check(id),cgh,sycl::write_only, sycl::no_init};
                        sycl::accessor U {shambase::get_check_ref(mpdat.pdat.get_field<Tscal>(iuint_interf).get_buf()),cgh,sycl::read_only};
                        sycl::accessor h {shambase::get_check_ref(mpdat.pdat.get_field<Tscal>(ihpart_interf).get_buf()),cgh,sycl::read_only};

                        Tscal pmass = gpart_mass;
                        Tscal gamma = this->eos_gamma;

                        cgh.parallel_for(sycl::range<1>{mpdat.total_elements},[=](sycl::item<1> item) {

                            using namespace shamrock::sph;
                            P[item] =  (gamma-1) * rho_h(pmass, h[item]) *U[item]  ; 

                        });

                    });

                });
            }

            //compute tree hmax
            //the smoothing won't change between steps so we can compute this only once
            if(corrector_iter_cnt == 0){

                rtree_field_h = trees.template map<RTreeField>(
                    [&](u64 id, RTree & rtree){
                        return rtree.compute_int_boxes(
                            shamsys::instance::get_compute_queue(), 
                            mpdat.get(id).pdat.get_field<Tscal>(ihpart_interf).get_buf(), 
                            1);
                    });
        
                
                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                    neigh_caches.preload(cur_p.id_patch);
                });
                
            }


            


            // compute force


            logger::info_ln("sph::BasicGas", "compute force");
            


            // save old acceleration
            logger::info_ln("sph::BasicGas", "save old fields");
            ComputeField<Tvec> old_axyz  = utility.save_field<Tvec>(iaxyz, "axyz_old");
            ComputeField<Tscal> old_duint = utility.save_field<Tscal>(iduint, "duint_old");



            if(enable_physics){



                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                    MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                    sycl::buffer<Tvec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(cur_p.id_patch).field.get_buf());
                    sycl::buffer<Tvec> & buf_axyz     = shambase::get_check_ref(pdat.get_field<Tvec>(iaxyz).get_buf());
                    sycl::buffer<Tscal> & buf_duint    = shambase::get_check_ref(pdat.get_field<Tscal>(iduint).get_buf());
                    sycl::buffer<Tvec> & buf_vxyz     = shambase::get_check_ref(merged_patch.pdat.get_field<Tvec>(ivxyz_interf).get_buf());
                    sycl::buffer<Tscal> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(ihpart_interf).get_buf());
                    sycl::buffer<Tscal> & buf_omega    = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(iomega_interf).get_buf());
                    sycl::buffer<Tscal> & buf_uint     = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(iuint_interf).get_buf());
                    sycl::buffer<Tscal> & buf_pressure = pressure.get_buf_check(cur_p.id_patch);

                    sycl::buffer<Tscal> & tree_field_hmax = shambase::get_check_ref(rtree_field_h.get(cur_p.id_patch).radix_tree_field_buf);

                    sycl::range range_npart{pdat.get_obj_cnt()};

                    RTree & tree = trees.get(cur_p.id_patch);


                    tree::ObjectCache & pcache = neigh_caches.get_cache(cur_p.id_patch);


                    /////////////////////////////////////////////

                    {
                        NamedStackEntry tmppp{"force compute"};
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                        const Tscal pmass = gpart_mass;
                        const Tscal gamma = this->eos_gamma;
                        const Tscal alpha_u = 1.0;
                        const Tscal alpha_AV = 1.0;
                        const Tscal beta_AV = 2.0;

                        //tree::ObjectIterator particle_looper(tree,cgh);

                        //tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

                        tree::ObjectCacheIterator particle_looper(pcache,cgh);

                        sycl::accessor xyz      {buf_xyz     , cgh, sycl::read_only}; 
                        sycl::accessor axyz     {buf_axyz    , cgh, sycl::write_only}; 
                        sycl::accessor du       {buf_duint   , cgh, sycl::write_only}; 
                        sycl::accessor vxyz     {buf_vxyz    , cgh, sycl::read_only}; 
                        sycl::accessor hpart    {buf_hpart   , cgh, sycl::read_only}; 
                        sycl::accessor omega    {buf_omega   , cgh, sycl::read_only}; 
                        sycl::accessor u        {buf_uint    , cgh, sycl::read_only}; 
                        sycl::accessor pressure {buf_pressure, cgh, sycl::read_only}; 

                        sycl::accessor hmax_tree {tree_field_hmax, cgh, sycl::read_only};

                            //sycl::stream out {4096,1024,cgh};

                        constexpr Tscal Rker2 = Kernel::Rkern*Kernel::Rkern;

                        cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                            u32 id_a = (u32)item.get_id(0);

                            using namespace shamrock::sph;

                            Tvec sum_axyz = {0, 0, 0};
                            Tscal sum_du_a = 0;
                            Tscal h_a        = hpart[id_a];


                            Tvec xyz_a = xyz[id_a];
                            Tvec vxyz_a = vxyz[id_a];

                            Tscal rho_a    = rho_h(pmass, h_a);
                            Tscal rho_a_sq = rho_a * rho_a;
                            Tscal rho_a_inv = 1./rho_a;

                            Tscal P_a     = pressure[id_a];
                            //f32 P_a     = cs * cs * rho_a;
                            Tscal omega_a = omega[id_a];

                            Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                            const Tscal u_a = u[id_a];

                            Tscal lambda_viscous_heating = 0.0;
                            Tscal lambda_conductivity = 0.0;
                            Tscal lambda_shock = 0.0;

                            Tscal cs_a = sycl::sqrt(gamma*P_a/rho_a);
                        
                            Tvec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            Tvec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            //particle_looper.rtree_for([&](u32 node_id, vec bmin,vec bmax) -> bool {
                            //    flt int_r_max_cell     = hmax_tree[node_id] * Kernel::Rkern;
                            //
                            //    using namespace walker::interaction_crit;
                            //
                            //    return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, bmin,
                            //                            bmax, int_r_max_cell);
                            //},[&](u32 id_b){
                            
                            particle_looper.for_each_object(id_a,[&](u32 id_b){
                                // compute only omega_a
                                Tvec dr = xyz_a - xyz[id_b];
                                Tscal rab2 = sycl::dot(dr,dr);
                                Tscal h_b  = hpart[id_b];

                                if (rab2 > h_a*h_a * Rker2 && rab2 > h_b*h_b * Rker2){
                                    return;
                                }

                                Tscal rab  = sycl::sqrt(rab2);
                                Tvec vxyz_b = vxyz[id_b]; 
                                Tvec v_ab = vxyz_a - vxyz_b;
                                const Tscal u_b = u[id_b];

                                Tvec r_ab_unit = dr / rab;

                                if (rab < 1e-9) {
                                    r_ab_unit = {0, 0, 0};
                                }

                                Tscal rho_b   = rho_h(pmass, h_b);
                                Tscal P_b     = pressure[id_b];
                                //f32 P_b     = cs * cs * rho_b;
                                Tscal omega_b = omega[id_b];
                                Tscal cs_b = sycl::sqrt(gamma*P_b/rho_b); 
                                Tscal v_ab_r_ab = sycl::dot(v_ab,r_ab_unit);
                                Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                                /////////////////
                                //internal energy update
                                // scalar : f32  | vector : f32_3
                                const Tscal alpha_a = alpha_AV; 
                                const Tscal alpha_b = alpha_AV;
                                Tscal vsig_a = alpha_a*cs_a + beta_AV*abs_v_ab_r_ab; 
                                Tscal vsig_b = alpha_b*cs_b + beta_AV*abs_v_ab_r_ab;
                                Tscal vsig_u =  abs_v_ab_r_ab;

                                Tscal dWab_a = Kernel::dW(rab, h_a);
                                Tscal dWab_b = Kernel::dW(rab, h_b);

                                //auto v_sig_a = alpha_AV * cs_a + beta_AV * sycl::distance(v_ab, dr);
                                lambda_viscous_heating +=  pmass * vsig_a * Tscal(0.5) * (sycl::pown(sycl::dot(v_ab, dr), 2) * dWab_a);
                                lambda_conductivity += pmass * alpha_u * vsig_u * (u_a - u_b)* Tscal(0.5) * (dWab_a * omega_a_rho_a_inv + dWab_b / (rho_b * omega_b));
                                sum_du_a += pmass * v_ab_r_ab * dWab_a;

                                //out << sum_du_a << "\n";
                                /////////////////

                                Tscal qa_ab = shambase::sycl_utils::g_sycl_max(- Tscal(0.5)*rho_a*vsig_a*v_ab_r_ab,Tscal(0)); 
                                Tscal qb_ab = shambase::sycl_utils::g_sycl_max(- Tscal(0.5)*rho_b*vsig_b*v_ab_r_ab,Tscal(0));

                                Tvec tmp = sph_pressure_symetric_av<Tvec, Tscal>(pmass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
                                                                    omega_b, qa_ab, qb_ab, r_ab_unit * dWab_a,
                                                                    r_ab_unit * dWab_b);

                                //logger::raw(shambase::format("pmass {}, rho_a {}, P_a {}, omega_a {}\n", pmass,rho_a, P_a, omega_a));
                                
                                //out << "add : " << tmp << "\n";

                                sum_axyz += tmp;
                            });
                                    
                            sum_du_a = P_a *rho_a_inv*omega_a_rho_a_inv * sum_du_a;
                            lambda_viscous_heating = - omega_a_rho_a_inv * lambda_viscous_heating;
                            lambda_shock = lambda_viscous_heating + lambda_conductivity;
                            sum_du_a = sum_du_a + lambda_shock;

                            // out << "sum : " << sum_axyz << "\n";

                            axyz[id_a] = sum_axyz;
                            du[id_a] = sum_du_a;

                        });
                    });
                    }
                
                });
                



                ComputeField<Tscal> vepsilon_v_sq = utility.make_compute_field<Tscal>("vmean epsilon_v^2", 1);
                ComputeField<Tscal> uepsilon_u_sq = utility.make_compute_field<Tscal>("umean epsilon_u^2", 1);

                // corrector
                logger::info_ln("sph::BasicGas", "leapfrog corrector");
                utility.fields_leapfrog_corrector<Tvec>(ivxyz, iaxyz, old_axyz, vepsilon_v_sq, dt / 2);
                utility.fields_leapfrog_corrector<Tscal>(iuint, iduint, old_duint, uepsilon_u_sq, dt / 2);
                
                Tscal rank_veps_v = sycl::sqrt(vepsilon_v_sq.compute_rank_max());
                ///////////////////////////////////////////
                // compute means //////////////////////////
                ///////////////////////////////////////////

                Tscal sum_vsq = utility.compute_rank_dot_sum<Tvec>(ivxyz);

                Tscal vmean_sq = shamalgs::collective::allreduce_sum(sum_vsq) / Tscal(Npart_all);

                Tscal vmean = sycl::sqrt(vmean_sq);
                

                Tscal rank_eps_v = rank_veps_v / vmean;


                Tscal eps_v = shamalgs::collective::allreduce_max(rank_eps_v);

                logger::info_ln("BasicGas", "epsilon v :",eps_v);

                if(eps_v > 1e-2){
                    logger::warn_ln("BasicGasSPH", 
                        shambase::format("the corrector tolerance are broken the step will be re rerunned\n    eps_v = {}",
                        eps_v));
                    need_rerun_corrector = true;

                    logger::info_ln("rerun corrector ...");
                }else{
                    need_rerun_corrector = false;
                }

                if(!need_rerun_corrector){

                    logger::info_ln("BasicGas", "computing next CFL");

                    ComputeField<Tscal> vsig_max_dt = utility.make_compute_field<Tscal>("vsig_a", 1);

                    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                        MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                        sycl::buffer<Tvec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(cur_p.id_patch).field.get_buf());
                        sycl::buffer<Tvec> & buf_vxyz     = shambase::get_check_ref(merged_patch.pdat.get_field<Tvec>(ivxyz_interf).get_buf());
                        sycl::buffer<Tscal> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(ihpart_interf).get_buf());
                        sycl::buffer<Tscal> & buf_uint     = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(iuint_interf).get_buf());
                        sycl::buffer<Tscal> & buf_pressure = pressure.get_buf_check(cur_p.id_patch);
                        sycl::buffer<Tscal> & vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);

                        sycl::range range_npart{pdat.get_obj_cnt()};

                        tree::ObjectCache & pcache = neigh_caches.get_cache(cur_p.id_patch);

                        /////////////////////////////////////////////

                        {
                            NamedStackEntry tmppp{"compute vsig"};
                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                            const Tscal pmass = gpart_mass;
                            const Tscal gamma = this->eos_gamma;
                            const Tscal alpha_u = 1.0;
                            const Tscal alpha_AV = 1.0;
                            const Tscal beta_AV = 2.0;


                            tree::ObjectCacheIterator particle_looper(pcache,cgh);

                            sycl::accessor xyz      {buf_xyz     , cgh, sycl::read_only}; 
                            sycl::accessor vxyz     {buf_vxyz    , cgh, sycl::read_only}; 
                            sycl::accessor hpart    {buf_hpart   , cgh, sycl::read_only}; 
                            sycl::accessor u        {buf_uint    , cgh, sycl::read_only}; 
                            sycl::accessor pressure {buf_pressure, cgh, sycl::read_only}; 
                            sycl::accessor vsig {vsig_buf, cgh, sycl::write_only,sycl::no_init};

                            constexpr Tscal Rker2 = Kernel::Rkern*Kernel::Rkern;

                            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                                u32 id_a = (u32)item.get_id(0);

                                using namespace shamrock::sph;

                                Tvec sum_axyz = {0, 0, 0};
                                Tscal sum_du_a = 0;
                                Tscal h_a        = hpart[id_a];


                                Tvec xyz_a = xyz[id_a];
                                Tvec vxyz_a = vxyz[id_a];

                                Tscal rho_a    = rho_h(pmass, h_a);
                                Tscal rho_a_sq = rho_a * rho_a;
                                Tscal rho_a_inv = 1./rho_a;

                                Tscal P_a     = pressure[id_a];

                                const Tscal u_a = u[id_a];

                                Tscal cs_a = sycl::sqrt(gamma*P_a/rho_a);

                                Tscal vsig_max = 0;
                                
                                particle_looper.for_each_object(id_a,[&](u32 id_b){
                                    // compute only omega_a
                                    Tvec dr = xyz_a - xyz[id_b];
                                    Tscal rab2 = sycl::dot(dr,dr);
                                    Tscal h_b  = hpart[id_b];

                                    if (rab2 > h_a*h_a * Rker2 && rab2 > h_b*h_b * Rker2){
                                        return;
                                    }

                                    Tscal rab  = sycl::sqrt(rab2);
                                    Tvec vxyz_b = vxyz[id_b]; 
                                    Tvec v_ab = vxyz_a - vxyz_b;
                                    const Tscal u_b = u[id_b];

                                    Tvec r_ab_unit = dr / rab;

                                    if (rab < 1e-9) {
                                        r_ab_unit = {0, 0, 0};
                                    }

                                    Tscal rho_b   = rho_h(pmass, h_b);
                                    Tscal P_b     = pressure[id_b];
                                    Tscal cs_b = sycl::sqrt(gamma*P_b/rho_b); 
                                    Tscal v_ab_r_ab = sycl::dot(v_ab,r_ab_unit);
                                    Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                                    /////////////////
                                    //internal energy update
                                    // scalar : f32  | vector : f32_3
                                    const Tscal alpha_a = alpha_AV; 
                                    const Tscal alpha_b = alpha_AV;

                                    Tscal vsig_a = alpha_a*cs_a + beta_AV*abs_v_ab_r_ab; 

                                    vsig_max = sycl::fmax(vsig_max, vsig_a);

                                });

                                vsig[id_a] = vsig_max;

                            });
                        });
                        }
                    
                    });

                    ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", 1);


                    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                        MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                        sycl::buffer<Tvec> & buf_axyz     = shambase::get_check_ref(pdat.get_field<Tvec>(iaxyz).get_buf());
                        sycl::buffer<Tscal> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<Tscal>(ihpart_interf).get_buf());
                        sycl::buffer<Tscal> & vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);
                        sycl::buffer<Tscal> & cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                            sycl::accessor hpart    {buf_hpart   , cgh, sycl::read_only}; 
                            sycl::accessor a    {buf_axyz   , cgh, sycl::read_only}; 
                            sycl::accessor vsig    {vsig_buf   , cgh, sycl::read_only}; 
                            sycl::accessor cfl_dt {cfl_dt_buf,cgh,sycl::write_only,sycl::no_init};

                            Tscal C_cour = cfl_cour;
                            Tscal C_force = cfl_force;

                            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                                Tscal h_a = hpart[item];
                                Tscal vsig_a = vsig[item];
                                Tscal abs_a_a = sycl::length(a[item]);

                                Tscal dt_c =  C_cour*h_a/vsig_a;
                                Tscal dt_f = C_force*sycl::sqrt(h_a/abs_a_a);

                                cfl_dt[item] = sycl::min(dt_c,dt_f);

                            });
                        });

                    });

                    Tscal rank_dt = cfl_dt.compute_rank_min();

                    logger::info_ln("BasigGas", "rank",shamsys::instance::world_rank,"found cfl dt =",rank_dt);


                    next_cfl = shamalgs::collective::allreduce_min(rank_dt);

                }


                corrector_iter_cnt ++;

            }



        }while(need_rerun_corrector);

        // if delta too big jump to compute force


        ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho",1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_h{
                    shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf()),
                    cgh,
                    sycl::read_only};

                sycl::accessor acc_rho{
                    shambase::get_check_ref(density.get_buf(p.id_patch)),
                    cgh,
                    sycl::write_only, sycl::no_init};
                const Tscal part_mass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid     = (u32)item.get_id();
                    using namespace shamrock::sph;
                    Tscal rho_ha = rho_h(part_mass, acc_h[gid]);
                    acc_rho[gid] = rho_ha;
                });
            });
        });

        if (dump_opt.vtk_do_dump) {
            shamrock::LegacyVtkWritter writter =
                start_dump<Tvec>(scheduler(), dump_opt.vtk_dump_fname);
            writter.add_point_data_section();

            u32 fnum = 0;
            if (dump_opt.vtk_dump_patch_id) {
                fnum += 2;
            }
            fnum++;
            fnum++;
            fnum++;
            fnum++;
            fnum++;
            fnum++;

            writter.add_field_data_section(fnum);

            if (dump_opt.vtk_dump_patch_id) {
                vtk_dump_add_patch_id(scheduler(), writter);
                vtk_dump_add_worldrank(scheduler(), writter);
            }


            vtk_dump_add_field<Tscal>(scheduler(), writter, ihpart, "h");
            vtk_dump_add_field<Tscal>(scheduler(), writter, iuint, "u");
            vtk_dump_add_field<Tvec>(scheduler(), writter, ivxyz, "v");
            vtk_dump_add_field<Tvec>(scheduler(), writter, iaxyz, "a");

            vtk_dump_add_compute_field(scheduler(), writter, density, "rho");
            vtk_dump_add_compute_field(scheduler(), writter, omega, "omega");
        }


        tstep.end();

        f64 rate = f64(scheduler().get_rank_count()) / tstep.elasped_sec();

        logger::info_ln("SPHSolver", "process rate : ",rate,"particle.s-1");

        return next_cfl;
}

using namespace shamrock::sph::kernels;

template class shammodels::SPHModelSolver<f64_3, M4>;