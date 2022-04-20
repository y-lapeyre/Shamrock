#pragma once

#include "aliases.hpp"
#include "hipSYCL/sycl/buffer.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "particles/particle_patch_mover.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/sphpatch.hpp"
#include "tree/radix_tree.hpp"
#include <memory>
#include <tuple>
#include <unordered_map>




template <class flt,class DataLayoutU3>
inline void leapfrog_predictor(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3) {

    using vec3 = sycl::vec<flt, 3>;

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->get_access<sycl::access::mode::read_write>(cgh);
        auto U3  = buf_U3->get_access<sycl::access::mode::read_write>(cgh);

        constexpr u32 nvar_U3 = DataLayoutU3::nvar;
        constexpr u32 ivxyz = DataLayoutU3::ivxyz;
        constexpr u32 iaxyz = DataLayoutU3::iaxyz;

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 & vxyz = U3[gid*nvar_U3 + ivxyz];
            vec3 & axyz = U3[gid*nvar_U3 + iaxyz];

            // v^{n + 1/2} = v^n + dt/2 a^n
            vxyz = vxyz + (dt / 2) * (axyz);

            // r^{n + 1} = r^n + dt v^{n + 1/2}
            xyz[gid] = xyz[gid] + dt * vxyz;

            // v^* = v^{n + 1/2} + dt/2 a^n
            vxyz = vxyz + (dt / 2) * (axyz);
        });
    };

    queue.submit(ker_predict_step);
}

template <class flt,class DataLayoutU3>
inline void leapfrog_corrector(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3){

    sycl::range<1> range_npart{npart};

    using vec3 = sycl::vec<flt, 3>;

    auto ker_corect_step = [&](sycl::handler &cgh) {
            

        auto U3  = buf_U3->get_access<sycl::access::mode::read_write>(cgh);

        constexpr u32 nvar_U3 = DataLayoutU3::nvar;
        constexpr u32 ivxyz = DataLayoutU3::ivxyz;
        constexpr u32 iaxyz = DataLayoutU3::iaxyz;
        constexpr u32 iaxyz_old = DataLayoutU3::iaxyz_old;


        // Executing kernel
        cgh.parallel_for(
            range_npart, [=](sycl::item<1> item) {
                
                u32 gid = (u32) item.get_id();

                vec3 & vxyz = U3[gid*nvar_U3 + ivxyz];
                vec3 & axyz = U3[gid*nvar_U3 + iaxyz];
                vec3 & axyz_old = U3[gid*nvar_U3 + iaxyz_old];
    
                //v^* = v^{n + 1/2} + dt/2 a^n
                vxyz = vxyz + (dt/2) * (axyz - axyz_old);

            }
        );

    };

    queue.submit(ker_corect_step);

}

template<class flt, class DataLayoutU3>
inline void swap_a_field(sycl::queue &queue, u32 npart,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3){
    sycl::range<1> range_npart{npart};

    using vec3 = sycl::vec<flt, 3>;

    auto ker_swap_a = [&](sycl::handler &cgh) {
            

        auto U3  = buf_U3->get_access<sycl::access::mode::read_write>(cgh);

        constexpr u32 nvar_U3 = DataLayoutU3::nvar;
        constexpr u32 iaxyz = DataLayoutU3::iaxyz;
        constexpr u32 iaxyz_old = DataLayoutU3::iaxyz_old;


        // Executing kernel
        cgh.parallel_for(
            range_npart, [=](sycl::item<1> item) {
                
                u32 gid = (u32) item.get_id();

                vec3 axyz = U3[gid*nvar_U3 + iaxyz];
                vec3 axyz_old = U3[gid*nvar_U3 + iaxyz_old];
    
                U3[gid*nvar_U3 + iaxyz] = axyz_old;
                U3[gid*nvar_U3 + iaxyz_old] = axyz;

            }
        );

    };

    queue.submit(ker_swap_a);
}



template<class DataLayout>
class SPHTimestepperLeapfrog{

    using pos_vec = typename DataLayout::pos;
    using pos_prec = typename pos_vec::T;

    using u_morton = u32;

    

    inline void step(SchedulerMPI &sched){
        SyCLHandler &hndl = SyCLHandler::get_instance();


        SerialPatchTree<pos_vec> sptree(sched.patch_tree, sched.get_box_tranform<pos_vec>());
        sptree.attach_buf();


        //cfl

        f32 dt_cur = 0.1f;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> leapfrog predictor" << std::endl;

            leapfrog_predictor<pos_prec, DataLayout::template U3<pos_prec>>(
                hndl.get_queue_compute(0), 
                pdat_buf.element_count, 
                dt_cur, 
                pdat_buf.get_pos<pos_vec>(), 
                pdat_buf.get_U3<pos_vec>());

            std::cout << "patch : n°"<<id_patch << " -> a field swap" << std::endl;

            swap_a_field<pos_prec, DataLayout::template U3<pos_prec>>(
                hndl.get_queue_compute(0), 
                pdat_buf.element_count, 
                pdat_buf.get_U3<pos_vec>());

        });

        std::cout << "particle reatribution" << std::endl;
        reatribute_particles(sched, sptree);







        pos_prec htol_up_tol = 1.2;


        std::cout << "exhanging interfaces" << std::endl;
        auto timer_h_max = timings::start_timer("compute_hmax", timings::timingtype::function);

        PatchField<f32> h_field;
        sched.compute_patch_field(
            h_field, 
            get_mpi_type<pos_prec>(), 
            [](sycl::queue & queue, Patch & p, PatchDataBuffer & pdat_buf){
                return patchdata::sph::get_h_max<DataLayout, pos_prec>(queue, pdat_buf)*htol_up_tol;
            }
        );

        timer_h_max.stop();

        InterfaceHandler<pos_vec, pos_prec> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<pos_vec, pos_prec>>(sched, sptree, h_field);
        interface_hndl.comm_interfaces(sched);







        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, pos_vec>>> radix_trees;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> making radix tree" << std::endl;

            //radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, pos_vec>>(hndl.get_queue_compute(0), sched.patch_data.sim_box.get_box<pos_prec>(cur_p), pdat_buf.get_pos<pos_vec>());
            

        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;

            radix_trees[id_patch].compute_cellvolume(hndl.get_queue_compute(0));

        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;

            radix_trees[id_patch].compute_int_boxes<
                DataLayout::template U1<pos_prec>::nvar,
                DataLayout::template U1<pos_prec>::ihpart
                >(hndl.get_queue_compute(0),pdat_buf.get_U1<pos_prec>());

        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°" << id_patch << " -> iterate h" << std::endl;

            std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, pos_vec>>> interface_trees;
            
            interface_hndl.for_each_interface(id_patch, hndl.get_queue_compute(0), [&hndl, &sched, &interface_trees](u64 patch_id, u64 interf_patch_id, PatchDataBuffer & interfpdat, std::tuple<f32_3,f32_3> box){

                std::cout << "  - interface : "<<interf_patch_id << " making tree" << std::endl;

                
                interface_trees[interf_patch_id] = std::make_unique<Radix_Tree<u_morton, pos_vec>>(hndl.get_queue_compute(0), box, interfpdat.get_pos<pos_vec>());


            });
            

        });





        

        
    }




};