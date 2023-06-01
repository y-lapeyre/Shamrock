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
#include "shamalgs/memory/memory.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/BasicSPHGhosts.hpp"
#include "shamrock/sph/SPHUtilities.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/tree/TreeTaversalCache.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/sph/forces.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shambase/time.hpp"











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

    



    f64 BasicGas::evolve(f64 dt, bool enable_physics, DumpOption dump_opt) {

        logger::info_ln("sph::BasicGas", ">>> Step :", dt);

        shambase::Timer tstep;
        tstep.start();

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


        u64 Npart_all = count_particles();

        using SPHUtils = SPHUtilities<vec, Kernel>;
        SPHUtils sph_utils(scheduler());

        BasicSPHGhostHandler<vec> interf_handle (scheduler());
        InterfacesUtility interf_utils(scheduler());
        
        auto interf_build_cache = sph_utils.build_interf_cache(interf_handle,sptree,htol_up_tol);
        auto merged_xyz = interf_handle.build_comm_merge_positions(interf_build_cache);

        constexpr u32 reduc_level = 3;

        using RTree = RadixTree<u_morton, vec>;

        shambase::DistributedData<RTree> trees = solver.make_merge_patch_trees(merged_xyz,reduc_level);

        ComputeField<flt> _epsilon_h = utility.make_compute_field<flt>("epsilon_h", 1,flt(100));
        ComputeField<flt> _h_old = utility.save_field<flt>(ihpart, "h_old");


        
        tree::ObjectCacheHandler hiter_caches{
            u64(10e9),
            [&](u64 patch_id){
                logger::debug_ln("SPHLeapfrog","patch : n°",patch_id,"->","gen cache");

                sycl::buffer<vec> & merged_r = shambase::get_check_ref(merged_xyz.get(patch_id).field.get_buf());
                sycl::buffer<flt> & hold = shambase::get_check_ref(_h_old.get_buf(patch_id));

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

                sycl::buffer<flt> & eps_h = shambase::get_check_ref(_epsilon_h.get_buf(p.id_patch));
                sycl::buffer<flt> & hold = shambase::get_check_ref(_h_old.get_buf(p.id_patch));

                sycl::buffer<flt> & hnew = shambase::get_check_ref(pdat.get_field<flt>(ihpart).get_buf());
                sycl::buffer<vec> & merged_r = shambase::get_check_ref(merged_xyz.get(p.id_patch).field.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree & tree = trees.get(p.id_patch);

                tree::ObjectCache & neigh_cache = hiter_caches.get_cache(p.id_patch);

                sph_utils.iterate_smoothing_lenght_cache(merged_r, hnew, hold, eps_h, range_npart, neigh_cache, gpart_mass, htol_up_tol, htol_up_iter);
                //sph_utils.iterate_smoothing_lenght_tree(merged_r, hnew, hold, eps_h, range_npart, tree, gpart_mass, htol_up_tol, htol_up_iter);
                
            });
        }

        //// compute omega
        ComputeField<flt> omega = utility.make_compute_field<flt>("omega", 1);
        {
            NamedStackEntry stack_loc2 {"compute omega"};

            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData & pdat){
                logger::debug_ln("SPHLeapfrog","patch : n°",p.id_patch,"->","h iteration");

                sycl::buffer<flt> & omega_h = shambase::get_check_ref(omega.get_buf(p.id_patch));

                sycl::buffer<flt> & hnew = shambase::get_check_ref(pdat.get_field<flt>(ihpart).get_buf());
                sycl::buffer<vec> & merged_r = shambase::get_check_ref(merged_xyz.get(p.id_patch).field.get_buf());

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
        interf_layout.add_field<flt>("hpart", 1);
        interf_layout.add_field<flt>("uint", 1);
        interf_layout.add_field<vec>("vxyz", 1);
        interf_layout.add_field<flt>("omega", 1);

        u32 ihpart_interf = 0;
        u32 iuint_interf  = 1;
        u32 ivxyz_interf  = 2;
        u32 iomega_interf = 3;

        

        using RTreeField = RadixTreeField<flt>;
        shambase::DistributedData<RTreeField> rtree_field_h;

        shambase::DistributedData<MergedPatchData> mpdat;

        tree::ObjectCacheHandler neigh_caches(u64(10e9),[&](u64 patch_id){

            logger::debug_ln("BasicSPH", "build particle cache id =",patch_id);

            NamedStackEntry cache_build_stack_loc{"build cache"};

            MergedPatchData & merged_patch = mpdat.get(patch_id);

            PatchData & pdat = scheduler().patch_data.get_pdat(patch_id);

            sycl::buffer<vec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(patch_id).field.get_buf());
            sycl::buffer<flt> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(ihpart_interf).get_buf());
            sycl::buffer<flt> & tree_field_hmax = shambase::get_check_ref(rtree_field_h.get(patch_id).radix_tree_field_buf);

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

        flt next_cfl = 0;

        
        u32 corrector_iter_cnt = 0;
        bool need_rerun_corrector = false;
        do{

            if(corrector_iter_cnt == 50){
                throw shambase::throw_with_loc<std::runtime_error>("the corrector has made over 50 loops, either their is a bug, either you are using a dt that is too large");
            }

                

            //communicate fields
            

            using InterfaceBuildInfos = BasicSPHGhostHandler<vec>::InterfaceBuildInfos;

            auto pdat_interf = interf_handle.build_interface_native<PatchData>(interf_build_cache, 
                [&](u64 sender,u64 /*receiver*/,InterfaceBuildInfos binfo, sycl::buffer<u32> & buf_idx, u32 cnt){

                    PatchData pdat(interf_layout);

                    PatchData & sender_patch = scheduler().patch_data.get_pdat(sender);
                    PatchDataField<flt> & sender_omega = omega.get_field(sender);

                    sender_patch.get_field<flt>(ihpart).append_subset_to(buf_idx,cnt,pdat.get_field<flt>(ihpart_interf));
                    sender_patch.get_field<flt>(iuint) .append_subset_to(buf_idx,cnt,pdat.get_field<flt>(iuint_interf));
                    sender_patch.get_field<vec>(ivxyz) .append_subset_to(buf_idx,cnt,pdat.get_field<vec>(ivxyz_interf));
                    sender_omega.append_subset_to(buf_idx,cnt,pdat.get_field<flt>(iomega_interf));
                    
                    pdat.check_field_obj_cnt_match();

                    return pdat;
                }
            );

            shambase::DistributedDataShared<PatchData> interf_pdat = 
                interf_handle.communicate_pdat(interf_layout, std::move(pdat_interf));


            mpdat = interf_handle.merge_native<PatchData,MergedPatchData>(
                std::move(interf_pdat), 
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchData & pdat){
                    
                    PatchData pdat_new(interf_layout);

                    u32 or_elem        = pdat.get_obj_cnt();
                    u32 total_elements = or_elem;

                    PatchDataField<flt> & cur_omega = omega.get_field(p.id_patch);

                    pdat_new.get_field<flt>(ihpart_interf).insert(pdat.get_field<flt>(ihpart));
                    pdat_new.get_field<flt>(iuint_interf ).insert(pdat.get_field<flt>(iuint));
                    pdat_new.get_field<vec>(ivxyz_interf ).insert(pdat.get_field<vec>(ivxyz));
                    pdat_new.get_field<flt>(iomega_interf).insert(cur_omega);
                    
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
            ComputeField<flt> pressure = utility.make_compute_field<flt>("pressure",1, [&](u64 id){
                return mpdat.get(id).total_elements;
            });

            {
                NamedStackEntry stack_loc{"compute eos"};
                
                mpdat.for_each([&](u64 id, MergedPatchData & mpdat){

                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor P {pressure.get_buf_check(id),cgh,sycl::write_only, sycl::no_init};
                        sycl::accessor U {shambase::get_check_ref(mpdat.pdat.get_field<flt>(iuint_interf).get_buf()),cgh,sycl::read_only};
                        sycl::accessor h {shambase::get_check_ref(mpdat.pdat.get_field<flt>(ihpart_interf).get_buf()),cgh,sycl::read_only};

                        flt pmass = gpart_mass;
                        flt gamma = this->gamma;

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

                rtree_field_h = trees.map<RTreeField>(
                    [&](u64 id, RTree & rtree){
                        return rtree.compute_int_boxes(
                            shamsys::instance::get_compute_queue(), 
                            mpdat.get(id).pdat.get_field<flt>(ihpart_interf).get_buf(), 
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
            ComputeField<vec> old_axyz  = utility.save_field<vec>(iaxyz, "axyz_old");
            ComputeField<flt> old_duint = utility.save_field<flt>(iduint, "duint_old");



            if(enable_physics){



                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                    MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                    sycl::buffer<vec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(cur_p.id_patch).field.get_buf());
                    sycl::buffer<vec> & buf_axyz     = shambase::get_check_ref(pdat.get_field<vec>(iaxyz).get_buf());
                    sycl::buffer<flt> & buf_duint    = shambase::get_check_ref(pdat.get_field<flt>(iduint).get_buf());
                    sycl::buffer<vec> & buf_vxyz     = shambase::get_check_ref(merged_patch.pdat.get_field<vec>(ivxyz_interf).get_buf());
                    sycl::buffer<flt> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(ihpart_interf).get_buf());
                    sycl::buffer<flt> & buf_omega    = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(iomega_interf).get_buf());
                    sycl::buffer<flt> & buf_uint     = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(iuint_interf).get_buf());
                    sycl::buffer<flt> & buf_pressure = pressure.get_buf_check(cur_p.id_patch);

                    sycl::buffer<flt> & tree_field_hmax = shambase::get_check_ref(rtree_field_h.get(cur_p.id_patch).radix_tree_field_buf);

                    sycl::range range_npart{pdat.get_obj_cnt()};

                    RTree & tree = trees.get(cur_p.id_patch);


                    tree::ObjectCache & pcache = neigh_caches.get_cache(cur_p.id_patch);


                    /////////////////////////////////////////////

                    {
                        NamedStackEntry tmppp{"force compute"};
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                        const flt pmass = gpart_mass;
                        const flt gamma = this->gamma;
                        const flt alpha_u = 1.0;
                        const flt alpha_AV = 1.0;
                        const flt beta_AV = 2.0;

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

                        constexpr flt Rker2 = Kernel::Rkern*Kernel::Rkern;

                        cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                            u32 id_a = (u32)item.get_id(0);

                            using namespace shamrock::sph;

                            vec sum_axyz = {0, 0, 0};
                            flt sum_du_a = 0;
                            flt h_a        = hpart[id_a];


                            vec xyz_a = xyz[id_a];
                            vec vxyz_a = vxyz[id_a];

                            flt rho_a    = rho_h(pmass, h_a);
                            flt rho_a_sq = rho_a * rho_a;
                            flt rho_a_inv = 1./rho_a;

                            flt P_a     = pressure[id_a];
                            //f32 P_a     = cs * cs * rho_a;
                            flt omega_a = omega[id_a];

                            flt omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                            const flt u_a = u[id_a];

                            flt lambda_viscous_heating = 0.0;
                            flt lambda_conductivity = 0.0;
                            flt lambda_shock = 0.0;

                            flt cs_a = sycl::sqrt(gamma*P_a/rho_a);
                        
                            vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

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
                                vec dr = xyz_a - xyz[id_b];
                                flt rab2 = sycl::dot(dr,dr);
                                flt h_b  = hpart[id_b];

                                if (rab2 > h_a*h_a * Rker2 && rab2 > h_b*h_b * Rker2){
                                    return;
                                }

                                flt rab  = sycl::sqrt(rab2);
                                vec vxyz_b = vxyz[id_b]; 
                                vec v_ab = vxyz_a - vxyz_b;
                                const flt u_b = u[id_b];

                                vec r_ab_unit = dr / rab;

                                if (rab < 1e-9) {
                                    r_ab_unit = {0, 0, 0};
                                }

                                flt rho_b   = rho_h(pmass, h_b);
                                flt P_b     = pressure[id_b];
                                //f32 P_b     = cs * cs * rho_b;
                                flt omega_b = omega[id_b];
                                flt cs_b = sycl::sqrt(gamma*P_b/rho_b); 
                                flt v_ab_r_ab = sycl::dot(v_ab,r_ab_unit);
                                flt abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                                /////////////////
                                //internal energy update
                                // scalar : f32  | vector : f32_3
                                const flt alpha_a = alpha_AV; 
                                const flt alpha_b = alpha_AV;
                                flt vsig_a = alpha_a*cs_a + beta_AV*abs_v_ab_r_ab; 
                                flt vsig_b = alpha_b*cs_b + beta_AV*abs_v_ab_r_ab;
                                flt vsig_u =  abs_v_ab_r_ab;

                                flt dWab_a = Kernel::dW(rab, h_a);
                                flt dWab_b = Kernel::dW(rab, h_b);

                                //auto v_sig_a = alpha_AV * cs_a + beta_AV * sycl::distance(v_ab, dr);
                                lambda_viscous_heating +=  pmass * vsig_a * flt(0.5) * (sycl::pown(sycl::dot(v_ab, dr), 2) * dWab_a);
                                lambda_conductivity += pmass * alpha_u * vsig_u * (u_a - u_b)* flt(0.5) * (dWab_a * omega_a_rho_a_inv + dWab_b / (rho_b * omega_b));
                                sum_du_a += pmass * v_ab_r_ab * dWab_a;

                                //out << sum_du_a << "\n";
                                /////////////////

                                flt qa_ab = shambase::sycl_utils::g_sycl_max(- flt(0.5)*rho_a*vsig_a*v_ab_r_ab,flt(0)); 
                                flt qb_ab = shambase::sycl_utils::g_sycl_max(- flt(0.5)*rho_b*vsig_b*v_ab_r_ab,flt(0));

                                vec tmp = sph_pressure_symetric_av<vec, flt>(pmass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
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
                



                ComputeField<flt> vepsilon_v_sq = utility.make_compute_field<flt>("vmean epsilon_v^2", 1);
                ComputeField<flt> uepsilon_u_sq = utility.make_compute_field<flt>("umean epsilon_u^2", 1);

                // corrector
                logger::info_ln("sph::BasicGas", "leapfrog corrector");
                utility.fields_leapfrog_corrector<vec>(ivxyz, iaxyz, old_axyz, vepsilon_v_sq, dt / 2);
                utility.fields_leapfrog_corrector<flt>(iuint, iduint, old_duint, uepsilon_u_sq, dt / 2);
                
                flt rank_veps_v = sycl::sqrt(vepsilon_v_sq.compute_rank_max());
                ///////////////////////////////////////////
                // compute means //////////////////////////
                ///////////////////////////////////////////

                flt sum_vsq = utility.compute_rank_dot_sum<vec>(ivxyz);

                flt vmean_sq = shamalgs::collective::allreduce_sum(sum_vsq) / flt(Npart_all);

                flt vmean = sycl::sqrt(vmean_sq);
                

                flt rank_eps_v = rank_veps_v / vmean;


                flt eps_v = shamalgs::collective::allreduce_max(rank_eps_v);

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

                    ComputeField<flt> vsig_max_dt = utility.make_compute_field<flt>("vsig_a", 1);

                    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                        MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                        sycl::buffer<vec> & buf_xyz      = shambase::get_check_ref(merged_xyz.get(cur_p.id_patch).field.get_buf());
                        sycl::buffer<vec> & buf_vxyz     = shambase::get_check_ref(merged_patch.pdat.get_field<vec>(ivxyz_interf).get_buf());
                        sycl::buffer<flt> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(ihpart_interf).get_buf());
                        sycl::buffer<flt> & buf_uint     = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(iuint_interf).get_buf());
                        sycl::buffer<flt> & buf_pressure = pressure.get_buf_check(cur_p.id_patch);
                        sycl::buffer<flt> & vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);

                        sycl::range range_npart{pdat.get_obj_cnt()};

                        tree::ObjectCache & pcache = neigh_caches.get_cache(cur_p.id_patch);

                        /////////////////////////////////////////////

                        {
                            NamedStackEntry tmppp{"compute vsig"};
                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                            const flt pmass = gpart_mass;
                            const flt gamma = this->gamma;
                            const flt alpha_u = 1.0;
                            const flt alpha_AV = 1.0;
                            const flt beta_AV = 2.0;


                            tree::ObjectCacheIterator particle_looper(pcache,cgh);

                            sycl::accessor xyz      {buf_xyz     , cgh, sycl::read_only}; 
                            sycl::accessor vxyz     {buf_vxyz    , cgh, sycl::read_only}; 
                            sycl::accessor hpart    {buf_hpart   , cgh, sycl::read_only}; 
                            sycl::accessor u        {buf_uint    , cgh, sycl::read_only}; 
                            sycl::accessor pressure {buf_pressure, cgh, sycl::read_only}; 
                            sycl::accessor vsig {vsig_buf, cgh, sycl::write_only,sycl::no_init};

                            constexpr flt Rker2 = Kernel::Rkern*Kernel::Rkern;

                            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                                u32 id_a = (u32)item.get_id(0);

                                using namespace shamrock::sph;

                                vec sum_axyz = {0, 0, 0};
                                flt sum_du_a = 0;
                                flt h_a        = hpart[id_a];


                                vec xyz_a = xyz[id_a];
                                vec vxyz_a = vxyz[id_a];

                                flt rho_a    = rho_h(pmass, h_a);
                                flt rho_a_sq = rho_a * rho_a;
                                flt rho_a_inv = 1./rho_a;

                                flt P_a     = pressure[id_a];

                                const flt u_a = u[id_a];

                                flt cs_a = sycl::sqrt(gamma*P_a/rho_a);

                                flt vsig_max = 0;
                                
                                particle_looper.for_each_object(id_a,[&](u32 id_b){
                                    // compute only omega_a
                                    vec dr = xyz_a - xyz[id_b];
                                    flt rab2 = sycl::dot(dr,dr);
                                    flt h_b  = hpart[id_b];

                                    if (rab2 > h_a*h_a * Rker2 && rab2 > h_b*h_b * Rker2){
                                        return;
                                    }

                                    flt rab  = sycl::sqrt(rab2);
                                    vec vxyz_b = vxyz[id_b]; 
                                    vec v_ab = vxyz_a - vxyz_b;
                                    const flt u_b = u[id_b];

                                    vec r_ab_unit = dr / rab;

                                    if (rab < 1e-9) {
                                        r_ab_unit = {0, 0, 0};
                                    }

                                    flt rho_b   = rho_h(pmass, h_b);
                                    flt P_b     = pressure[id_b];
                                    flt cs_b = sycl::sqrt(gamma*P_b/rho_b); 
                                    flt v_ab_r_ab = sycl::dot(v_ab,r_ab_unit);
                                    flt abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                                    /////////////////
                                    //internal energy update
                                    // scalar : f32  | vector : f32_3
                                    const flt alpha_a = alpha_AV; 
                                    const flt alpha_b = alpha_AV;

                                    flt vsig_a = alpha_a*cs_a + beta_AV*abs_v_ab_r_ab; 

                                    vsig_max = sycl::fmax(vsig_max, vsig_a);

                                });

                                vsig[id_a] = vsig_max;

                            });
                        });
                        }
                    
                    });

                    ComputeField<flt> cfl_dt = utility.make_compute_field<flt>("cfl_dt", 1);


                    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

                        MergedPatchData & merged_patch = mpdat.get(cur_p.id_patch);

                        sycl::buffer<vec> & buf_axyz     = shambase::get_check_ref(pdat.get_field<vec>(iaxyz).get_buf());
                        sycl::buffer<flt> & buf_hpart    = shambase::get_check_ref(merged_patch.pdat.get_field<flt>(ihpart_interf).get_buf());
                        sycl::buffer<flt> & vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);
                        sycl::buffer<flt> & cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                            sycl::accessor hpart    {buf_hpart   , cgh, sycl::read_only}; 
                            sycl::accessor a    {buf_axyz   , cgh, sycl::read_only}; 
                            sycl::accessor vsig    {vsig_buf   , cgh, sycl::read_only}; 
                            sycl::accessor cfl_dt {cfl_dt_buf,cgh,sycl::write_only,sycl::no_init};

                            flt C_cour = cfl_cour;
                            flt C_force = cfl_force;

                            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {

                                flt h_a = hpart[item];
                                flt vsig_a = vsig[item];
                                flt abs_a_a = sycl::length(a[item]);

                                flt dt_c =  C_cour*h_a/vsig_a;
                                flt dt_f = C_force*sycl::sqrt(h_a/abs_a_a);

                                cfl_dt[item] = sycl::min(dt_c,dt_f);

                            });
                        });

                    });

                    flt rank_dt = cfl_dt.compute_rank_min();

                    logger::info_ln("BasigGas", "rank",shamsys::instance::world_rank,"found cfl dt =",rank_dt);


                    next_cfl = shamalgs::collective::allreduce_min(rank_dt);

                }


                corrector_iter_cnt ++;

            }



        }while(need_rerun_corrector);

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
            fnum++;
            fnum++;

            writter.add_field_data_section(fnum);

            if (dump_opt.vtk_dump_patch_id) {
                vtk_dump_add_patch_id(scheduler(), writter);
                vtk_dump_add_worldrank(scheduler(), writter);
            }


            vtk_dump_add_field<flt>(scheduler(), writter, ihpart, "h");
            vtk_dump_add_field<flt>(scheduler(), writter, iuint, "u");
            vtk_dump_add_field<vec>(scheduler(), writter, ivxyz, "v");
            vtk_dump_add_field<vec>(scheduler(), writter, iaxyz, "a");

            vtk_dump_add_compute_field(scheduler(), writter, density, "rho");
            vtk_dump_add_compute_field(scheduler(), writter, omega, "omega");
        }


        tstep.end();

        f64 rate = f64(scheduler().get_rank_count()) / tstep.elasped_sec();

        logger::info_ln("BasicSPH", "process rate : ",rate,"particle.s-1");

        return next_cfl;
    }

} // namespace shammodels::sph