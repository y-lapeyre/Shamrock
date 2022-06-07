// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "algs/syclreduction.hpp"
#include "aliases.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "io/dump.hpp"
#include "io/logs.hpp"
#include "particles/particle_patch_mover.hpp"
#include "patch/compute_field.hpp"
#include "patch/global_var.hpp"
#include "patch/merged_patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/patchdata_field.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/kernels.hpp"
#include "sph/smoothing_lenght.hpp"
#include "sph/sphpart.hpp"
#include "sph/sphpatch.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "tree/radix_tree.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace integrators {

namespace sph {

    template <class flt,class Kernel,class u_morton> class LeapfrogGeneral {public:

    using vec3 = sycl::vec<flt, 3>;

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value || std::is_same<flt, f64>::value
    , "Leapfrog : floating point type should be one of (f16,f32,f64)");


    
    inline static void sycl_move_parts(sycl::queue &queue, u32 npart, flt dt, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
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



    inline static void sycl_position_modulo(sycl::queue &queue, u32 npart, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
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


























    //mandatory variables
    SchedulerMPI &sched;
    bool periodic_mode;
    flt htol_up_tol  ;
    flt htol_up_iter ;
    

    flt sph_gpart_mass;



    LeapfrogGeneral(SchedulerMPI &sched,bool periodic_mode,flt htol_up_tol,
        flt htol_up_iter ,flt sph_gpart_mass) : 
            sched(sched), 
            periodic_mode(periodic_mode) , 
            htol_up_tol(htol_up_tol) , 
            htol_up_iter(htol_up_iter),
            sph_gpart_mass(sph_gpart_mass)
        {}


    template<class LambdaCFL,class LambdaUpdateTime,class LambdaSwapDer, class LambdaForce, class LambdaCorrector>
    inline flt step(flt old_time, bool do_force, bool do_corrector,
        LambdaCFL && lambda_cfl,
        LambdaUpdateTime && lambda_update_time,
        LambdaSwapDer && lambda_swap_der,
        LambdaForce && lambda_compute_forces,
        LambdaCorrector && lambda_correct){

        SyCLHandler &hndl = SyCLHandler::get_instance();

        const flt loc_htol_up_tol  = htol_up_tol;
        const flt loc_htol_up_iter = htol_up_iter;

        const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

        const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");




        //Init serial patch tree
        SerialPatchTree<vec3> sptree(sched.patch_tree, sched.get_box_tranform<vec3>());
        sptree.attach_buf();


        //compute cfl
        auto get_cfl = [&]() -> flt{
            GlobalVariable<min,flt> cfl_glb_var;
            cfl_glb_var.compute_var_patch(sched, lambda_cfl);
            cfl_glb_var.reduce_val();
            return cfl_glb_var.get_val();
        };
        
        flt cfl_val = get_cfl();




        //compute dt step

        f32 dt_cur = sycl::min(f32(0.001),cfl_val);

        std::cout << " --- current dt  : " << dt_cur << std::endl;

        //advance time
        flt step_time = old_time;
        step_time += dt_cur;

        //leapfrog predictor
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer &pdat_buf) {
            std::cout << "patch : n°" << id_patch << " -> leapfrog predictor" << std::endl;

            lambda_update_time(hndl.get_queue_compute(0),pdat_buf,sycl::range<1> {pdat_buf.element_count},dt_cur/2);

            sycl_move_parts(hndl.get_queue_compute(0), pdat_buf.element_count, dt_cur,
                                              pdat_buf.fields_f32_3.at(ixyz), pdat_buf.fields_f32_3.at(ivxyz));

            lambda_update_time(hndl.get_queue_compute(0),pdat_buf,sycl::range<1> {pdat_buf.element_count},dt_cur/2);


            std::cout << "patch : n°" << id_patch << " -> a field swap" << std::endl;

            lambda_swap_der(hndl.get_queue_compute(0),pdat_buf,sycl::range<1> {pdat_buf.element_count});

            if (periodic_mode) {
                sycl_position_modulo(hndl.get_queue_compute(0), pdat_buf.element_count,
                                               pdat_buf.fields_f32_3[ixyz], sched.get_box_volume<vec3>());
            }
        });

        //move particles between patches
        std::cout << "particle reatribution" << std::endl;
        reatribute_particles(sched, sptree, periodic_mode);

        

        std::cout << "exhanging interfaces" << std::endl;
        auto timer_h_max = timings::start_timer("compute_hmax", timings::timingtype::function);


        //compute hmax
        PatchField<f32> h_field;
        sched.compute_patch_field(
            h_field, get_mpi_type<flt>(), [loc_htol_up_tol](sycl::queue &queue, Patch &p, PatchDataBuffer &pdat_buf) {
                return patchdata::sph::get_h_max<flt>(pdat_buf.pdl, queue, pdat_buf) * loc_htol_up_tol * Kernel::Rkern;
            });

        timer_h_max.stop();


        //make interfaces
        InterfaceHandler<vec3, flt> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<vec3, flt>>(sched, sptree, h_field,
                                                                                                 periodic_mode);
        interface_hndl.comm_interfaces(sched, periodic_mode);


        // merging strategy
        auto tmerge_buf = timings::start_timer("buffer merging", timings::sycl);
        std::unordered_map<u64, MergedPatchDataBuffer<vec3>> merge_pdat_buf;
        make_merge_patches(sched, interface_hndl, merge_pdat_buf);
        hndl.get_queue_compute(0).wait();
        tmerge_buf.stop();


        //make trees
        auto tgen_trees = timings::start_timer("radix tree compute", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, vec3>>> radix_trees;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°" << id_patch << " -> making radix tree" << std::endl;
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0)
                std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer &mpdat_buf = *merge_pdat_buf.at(id_patch).data;

            std::tuple<f32_3, f32_3> &box = merge_pdat_buf.at(id_patch).box;

            // radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, vec3>>(hndl.get_queue_compute(0), box,
                                                                                    mpdat_buf.fields_f32_3[ixyz]);
        });

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°" << id_patch << " -> radix tree compute volume" << std::endl;
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0)
                std::cout << " empty => skipping" << std::endl;
            radix_trees[id_patch]->compute_cellvolume(hndl.get_queue_compute(0));
        });

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°" << id_patch << " -> radix tree compute interaction box" << std::endl;
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0)
                std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer &mpdat_buf = *merge_pdat_buf.at(id_patch).data;

            radix_trees[id_patch]->compute_int_boxes(hndl.get_queue_compute(0), mpdat_buf.fields_f32[ihpart], htol_up_tol);
        });
        hndl.get_queue_compute(0).wait();
        tgen_trees.stop();


        //create compute field for new h and omega
        std::cout << "making omega field" << std::endl;
        PatchComputeField<f32> hnew_field;
        PatchComputeField<f32> omega_field;

        hnew_field.generate(sched);
        omega_field.generate(sched);

        hnew_field.to_sycl();
        omega_field.to_sycl();

        //iterate smoothing lenght
        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°" << id_patch << "init h iter" << std::endl;
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0)
                std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

            sycl::buffer<f32> &hnew  = *hnew_field.field_data_buf[id_patch];
            sycl::buffer<f32> &omega = *omega_field.field_data_buf[id_patch];
            sycl::buffer<f32> eps_h  = sycl::buffer<f32>(merge_pdat_buf.at(id_patch).or_element_cnt);

            sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

            std::cout << "   original size : " << merge_pdat_buf.at(id_patch).or_element_cnt
                      << " | merged : " << pdat_buf_merge.element_count << std::endl;

            ::sph::algs::SmoothingLenghtCompute<f32, u32, Kernel> h_iterator(sched.pdl, htol_up_tol, htol_up_iter);

            h_iterator.iterate_smoothing_lenght(hndl.get_queue_compute(0), merge_pdat_buf.at(id_patch).or_element_cnt,
                                                sph_gpart_mass, *radix_trees[id_patch], pdat_buf_merge, hnew, omega, eps_h);

            // write back h test
            //*
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);

                auto acc_hpart = pdat_buf_merge.fields_f32.at(ihpart)->get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class write_back_h>(range_npart,
                                                     [=](sycl::item<1> item) { acc_hpart[item] = h_new[item]; });
            });
            //*/
        });




        // exchange new h and omega
        hnew_field.to_map();
        omega_field.to_map();

        std::cout << "echange interface hnew" << std::endl;
        PatchComputeFieldInterfaces<flt> hnew_field_interfaces =
            interface_hndl.template comm_interfaces_field<flt>(sched, hnew_field, periodic_mode);
        std::cout << "echange interface omega" << std::endl;
        PatchComputeFieldInterfaces<flt> omega_field_interfaces =
            interface_hndl.template comm_interfaces_field<flt>(sched, omega_field, periodic_mode);

        hnew_field.to_sycl();
        omega_field.to_sycl();

        //merge compute fields
        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> hnew_field_merged;
        make_merge_patches_comp_field<f32>(sched, interface_hndl, hnew_field, hnew_field_interfaces, hnew_field_merged);
        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> omega_field_merged;
        make_merge_patches_comp_field<f32>(sched, interface_hndl, omega_field, omega_field_interfaces, omega_field_merged);

        //TODO add looping on corrector step


        //compute force
        if (do_force) {
            lambda_compute_forces(sched,radix_trees,merge_pdat_buf,hnew_field_merged,omega_field_merged,htol_up_tol);
        }



        //leapfrog corrector
        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0){
                std::cout << " empty => skipping" << std::endl;return;
            }

            PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

            if (do_corrector) {

                std::cout << "leapfrog corrector " << std::endl;

                

                lambda_correct(hndl.get_queue_compute(0),pdat_buf_merge,sycl::range<1> {merge_pdat_buf.at(id_patch).or_element_cnt},dt_cur/2);
            }
        });

        write_back_merge_patches(sched, interface_hndl, merge_pdat_buf);

        return step_time;
    }

};




}

}
