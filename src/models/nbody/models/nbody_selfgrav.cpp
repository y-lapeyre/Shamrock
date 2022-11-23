
#include "nbody_selfgrav.hpp"
#include "core/patch/interfaces/interface_handler.hpp"
#include "core/patch/utility/full_tree_field.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "core/tree/radix_tree.hpp"
#include "runscript/shamrockapi.hpp"

#include "models/generic/algs/integrators_utils.hpp"

#include "core/patch/comm/patch_object_mover.hpp"


const std::string console_tag = "[NBodySelfGrav] ";


template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::check_valid(){


    if (cfl_force < 0) {
        throw ShamAPIException(console_tag + "cfl force not set");
    }

    if (gpart_mass < 0) {
        throw ShamAPIException(console_tag + "particle mass not set");
    }
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::init(){

}









template<class flt,class vec3>
void sycl_move_parts(sycl::queue &queue, u32 npart, flt dt, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_vxyz) {

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto acc_xyz  = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_vxyz = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 &vxyz = acc_vxyz[item];

            acc_xyz[item] = acc_xyz[item] + dt * vxyz;

        });
    };

    queue.submit(ker_predict_step);
}


template<class vec3>
void sycl_position_modulo(sycl::queue &queue, u32 npart, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                                std::tuple<vec3, vec3> box) {

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);
        vec3 delt    = box_max - box_min;

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 r = xyz[gid] - box_min;

            r = sycl::fmod(r, delt);
            r += delt;
            r = sycl::fmod(r, delt);
            r += box_min;

            xyz[gid] = r;
        });
    };

    queue.submit(ker_predict_step);
}






template<class flt>
class FMMInteract_cd{

    using vec = sycl::vec<flt, 3>;

    flt opening_crit_sq;

    public:

    explicit FMMInteract_cd(flt open_crit) : opening_crit_sq(open_crit*open_crit){};

    static bool interact_cd_cell_cell(const FMMInteract_cd & cd,vec b1_min, vec b1_max,vec b2_min, vec b2_max){
        vec s1 = (b1_max + b1_min)/2;
        vec s2 = (b2_max + b2_min)/2;

        vec r_fmm = s2-s1;

        vec d1 = b1_max - b1_min;
        vec d2 = b2_max - b2_min;

        flt l1 = sycl::max(sycl::max(d1.x(),d1.y()),d1.z());
        flt l2 = sycl::max(sycl::max(d2.x(),d2.y()),d2.z());

        flt opening_angle_sq = (l1 + l2)*(l1 + l2)/sycl::dot(r_fmm,r_fmm);

        return opening_angle_sq < cd.opening_crit_sq;
    }

    static bool interact_cd_cell_patch(const FMMInteract_cd & cd,vec b1_min, vec b1_max,vec b2_min, vec b2_max, flt b1_min_slenght, flt b1_max_slenght, flt b2_min_slenght, flt b2_max_slenght){
        
        vec c1 = (b1_max + b1_min)/2;
        flt L1 = sycl::max(sycl::max(c1.x(),c1.y()),c1.z());

        flt dist_to_surf = sycl::sqrt(BBAA::get_sq_distance_to_BBAAsurface(c1, b2_min, b2_max));

        flt opening_angle_sq = (L1 + b2_max_slenght)/(dist_to_surf - b2_min_slenght/2);
        opening_angle_sq *= opening_angle_sq;

        return opening_angle_sq < cd.opening_crit_sq;
    }

    static bool interact_cd_cell_patch_outdomain(const FMMInteract_cd & cd,vec b1_min, vec b1_max,vec b2_min, vec b2_max, flt b1_min_slenght, flt b1_max_slenght, flt b2_min_slenght, flt b2_max_slenght){
        return false;
    }
};







template<class flt> 
f64 models::nbody::Nbody_SelfGrav<flt>::evolve(PatchScheduler &sched, f64 old_time, f64 target_time){

    check_valid();

    logger::info_ln("NBodySelfGrav", "evolve t=",old_time);


    //Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);

    const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
    const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
    const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
    const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

    //const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

    //PatchComputeField<f32> pressure_field;


    auto lambda_update_time = [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ,flt hdt){
            
        sycl::buffer<vec3> & vxyz =  * pdat.get_field<vec3>(ivxyz).get_buf();
        sycl::buffer<vec3> & axyz =  * pdat.get_field<vec3>(iaxyz).get_buf();

        field_advance_time(queue, vxyz, axyz, range_npart, hdt);

    };

    auto lambda_swap_der = [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ){
        auto ker_predict_step = [&](sycl::handler &cgh) {
            auto acc_axyz = pdat.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = pdat.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                vec3 axyz     = acc_axyz[item];
                vec3 axyz_old = acc_axyz_old[item];

                acc_axyz[item]     = axyz_old;
                acc_axyz_old[item] = axyz;

            });
        };

        queue.submit(ker_predict_step);
    };

    auto lambda_correct = [&](sycl::queue&  queue, PatchData& buf, sycl::range<1> range_npart ,flt hdt){
            
        auto ker_corect_step = [&](sycl::handler &cgh) {
            auto acc_vxyz     = buf.get_field<vec3>(ivxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz     = buf.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                //u32 gid = (u32)item.get_id();
                //
                //vec3 &vxyz     = acc_vxyz[item];
                //vec3 &axyz     = acc_axyz[item];
                //vec3 &axyz_old = acc_axyz_old[item];

                // v^* = v^{n + 1/2} + dt/2 a^n
                acc_vxyz[item] = acc_vxyz[item] + (hdt) * (acc_axyz[item] - acc_axyz_old[item]);
            });
        };

        queue.submit(ker_corect_step);
    };



    auto leapfrog_lambda = [&](flt old_time, bool do_force, bool do_corrector) -> flt {

        const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");



        logger::info_ln("NBodyleapfrog", "step t=",old_time, "do_force =",do_force, "do_corrector =",do_corrector);





        //Init serial patch tree
        SerialPatchTree<vec3> sptree(sched.patch_tree, sched.get_box_tranform<vec3>());
        sptree.attach_buf();

        //compute cfl
        flt cfl_val = 1e-3;




        //compute dt step

        flt dt_cur = cfl_val;

        logger::info_ln("SPHLeapfrog", "current dt  :",dt_cur);

        //advance time
        flt step_time = old_time;
        step_time += dt_cur;

        //leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {

            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","predictor");

            lambda_update_time(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);

            sycl_move_parts(sycl_handler::get_compute_queue(), pdat.get_obj_cnt(), dt_cur,
                                              pdat.get_field<vec3>(ixyz).get_buf(), pdat.get_field<vec3>(ivxyz).get_buf());

            lambda_update_time(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);


            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","dt fields swap");

            lambda_swap_der(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()});

            if (periodic_bc) {//TODO generalise position modulo in the scheduler
                sycl_position_modulo(sycl_handler::get_compute_queue(), pdat.get_obj_cnt(),
                                               pdat.get_field<vec3>(ixyz).get_buf(), sched.get_box_volume<vec3>());
            }
        });




        //move particles between patches
        logger::debug_ln("SPHLeapfrog", "particle reatribution");
        reatribute_particles(sched, sptree, periodic_bc);





        constexpr u32 reduc_level = 5;

        using RadTree = Radix_Tree<u_morton, vec3>;

        //make trees
        auto tgen_trees = timings::start_timer("radix tree gen", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<RadTree>> radix_trees;

        sched.for_each_patch_data([&](u64 id_patch, Patch & cur_p, PatchData & pdat) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","making Radix Tree");

            if (pdat.is_empty()){
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree build");
            }else{

                auto & buf_xyz = pdat.get_field<vec3>(ixyz).get_buf();

                std::tuple<vec3, vec3> box = sched.patch_data.sim_box.get_box<flt>(cur_p);

                // radix tree computation
                radix_trees[id_patch] = std::make_unique<RadTree>(sycl_handler::get_compute_queue(), box,
                                                                                    buf_xyz,pdat.get_obj_cnt(),reduc_level);
            }
                
        });


        sched.for_each_patch_data([&](u64 id_patch, Patch &  /*cur_p*/, PatchData & pdat) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","compute radix tree cell volumes");
            if (pdat.is_empty()){
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree build");
            }else{
                radix_trees[id_patch]->compute_cellvolume(sycl_handler::get_compute_queue());
            }
        });



        sycl_handler::get_compute_queue().wait();
        tgen_trees.stop();




        auto box = sched.get_box_tranform<vec3>();
        SimulationDomain<flt> sd(Free, std::get<0>(box), std::get<1>(box));


        







        //generate the tree field for the box size info

        PatchField<flt> max_slenght_cells;
        PatchField<flt> min_slenght_cells;

        std::unordered_map<u64, flt> min_slenght_map;
        std::unordered_map<u64, flt> max_slenght_map;

        for(auto & [k,rtree_ptr] : radix_trees){
            auto [min,max] = rtree_ptr->get_min_max_cell_side_lenght();
            min_slenght_map[k] = min;
            max_slenght_map[k] = max;
        }

        sched.compute_patch_field(
            min_slenght_cells, get_mpi_type<flt>(), [&](sycl::queue & /*queue*/, Patch &p, PatchData & /*pdat*/) {
                return min_slenght_map[p.id_patch];
            });

        sched.compute_patch_field(
            max_slenght_cells, get_mpi_type<flt>(), [&](sycl::queue & /*queue*/, Patch &p, PatchData & /*pdat*/) {
                return max_slenght_map[p.id_patch];
            });

        using RtreeField = typename RadTree::template RadixTreeField<flt>;
        std::unordered_map<u64, std::unique_ptr<RtreeField>> min_tree_slenght_map;
        std::unordered_map<u64, std::unique_ptr<RtreeField>> max_tree_slenght_map;

        for(auto & [k,rtree_ptr] : radix_trees){
            std::unique_ptr<RtreeField> & field_min = min_tree_slenght_map[k];
            std::unique_ptr<RtreeField> & field_max = max_tree_slenght_map[k];

            u32 total_cell_bount = rtree_ptr->tree_internal_count + rtree_ptr->tree_leaf_count;

            field_min = std::make_unique<RtreeField>();
            field_min->nvar = 1;
            field_min->radix_tree_field_buf = std::make_unique<sycl::buffer<flt>>(total_cell_bount);

            field_max = std::make_unique<RtreeField>();
            field_max->nvar = 1;
            field_max->radix_tree_field_buf = std::make_unique<sycl::buffer<flt>>(total_cell_bount);

            auto & buf_pos_min_cell_flt = rtree_ptr->buf_pos_min_cell_flt;
            auto & buf_pos_max_cell_flt = rtree_ptr->buf_pos_max_cell_flt;

            auto & rfield_buf_min = field_min->radix_tree_field_buf;
            auto & rfield_buf_max = field_max->radix_tree_field_buf;

            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor box_min_cell {*buf_pos_min_cell_flt, cgh,sycl::read_only};
                sycl::accessor box_max_cell {*buf_pos_max_cell_flt, cgh,sycl::read_only};

                sycl::accessor s_lengh_min {*rfield_buf_min, cgh,sycl::write_only,sycl::no_init};
                sycl::accessor s_lengh_max {*rfield_buf_max, cgh,sycl::write_only,sycl::no_init};

                cgh.parallel_for(sycl::range<1>{total_cell_bount}, [=](sycl::item<1> item) {

                    vec3 bmin = box_min_cell[item];
                    vec3 bmax = box_max_cell[item];

                    vec3 sz = bmax - bmin;

                    s_lengh_min[item] = sycl::fmin(sycl::fmin(sz.x(),sz.y()),sz.z());
                    s_lengh_max[item] = sycl::fmax(sycl::fmax(sz.x(),sz.y()),sz.z());

                });

            });

        }

        FullTreeField<flt, RadTree> min_slenght{
            std::move(min_slenght_cells),
            std::move(min_tree_slenght_map)
        };

        FullTreeField<flt, RadTree> max_slenght{
            std::move(min_slenght_cells),
            std::move(min_tree_slenght_map)
        };


        flt open_crit = 0.3;

        using InterfHndl =  Interfacehandler<Tree_Send, flt, RadTree>;
        InterfHndl interf_hndl = InterfHndl();
        interf_hndl.compute_interface_list(sched,sptree,sd,radix_trees,FMMInteract_cd<flt>(open_crit),min_slenght,max_slenght);


        
        

        //make interfaces



        //force



        //leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {

            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","corrector");

            lambda_correct(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);

        });


        return step_time;

    };
    







    f64 step_time = leapfrog_lambda(old_time,true,true);













    return step_time;
}


template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::dump(std::string prefix){
    std::cout << "dump : "<< prefix << std::endl;
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::restart_dump(std::string prefix){
    std::cout << "restart dump : "<< prefix << std::endl;
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::close(){
    
}



template class models::nbody::Nbody_SelfGrav<f32>;

