#pragma once

#include "aliases.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "io/logs.hpp"
#include "particles/particle_patch_mover.hpp"
#include "patch/compute_field.hpp"
#include "patch/patchdata.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/kernels.hpp"
#include "sph/sphpatch.hpp"
#include "tree/radix_tree.hpp"
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>




template <class flt,class DataLayoutU3>
inline void leapfrog_predictor(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3) {

    using vec3 = sycl::vec<flt, 3>;

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto U3  = buf_U3->template get_access<sycl::access::mode::read_write>(cgh);

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
            

        auto U3  = buf_U3->template get_access<sycl::access::mode::read_write>(cgh);

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
            

        auto U3  = buf_U3->template get_access<sycl::access::mode::read_write>(cgh);

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
class SPHTimestepperLeapfrog{public:

    
    using pos_prec = typename DataLayout::pos_type;
    using pos_vec  = sycl::vec<pos_prec, 3>;

    using u_morton = u32;

    using Kernel = sph::kernels::M4<f32>;

    using DU1 = typename DataLayout::template U1<pos_prec>::T;
    using DU3 = typename DataLayout::template U3<pos_prec>::T;

    inline void step(SchedulerMPI &sched){
        SyCLHandler &hndl = SyCLHandler::get_instance();


        SerialPatchTree<pos_vec> sptree(sched.patch_tree, sched.get_box_tranform<pos_vec>());
        sptree.attach_buf();


        //cfl

        f32 dt_cur = 0.1f;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> leapfrog predictor" << std::endl;

            leapfrog_predictor<pos_prec, DU3>(
                hndl.get_queue_compute(0), 
                pdat_buf.element_count, 
                dt_cur, 
                pdat_buf.get_pos<pos_vec>(), 
                pdat_buf.get_U3<pos_vec>());

            std::cout << "patch : n°"<<id_patch << " -> a field swap" << std::endl;

            swap_a_field<pos_prec, DU3>(
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
            [htol_up_tol](sycl::queue & queue, Patch & p, PatchDataBuffer & pdat_buf){
                return patchdata::sph::get_h_max<DataLayout, pos_prec>(queue, pdat_buf)*htol_up_tol;
            }
        );

        timer_h_max.stop();




        InterfaceHandler<pos_vec, pos_prec> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<pos_vec, pos_prec>>(sched, sptree, h_field);
        interface_hndl.comm_interfaces(sched);





        //merging strat
        auto tmerge_buf = timings::start_timer("buffer merging", timings::sycl);
        std::unordered_map<u64,PatchDataBuffer> merge_pdat_buf;
        std::unordered_map<u64,std::tuple<f32_3,f32_3>> merge_pdat_box;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            auto tmp_box = sched.patch_data.sim_box.get_box<pos_prec>(cur_p);

            f32_3 min_box = std::get<0>(tmp_box);
            f32_3 max_box = std::get<1>(tmp_box);

            std::cout << "patch : n°"<<id_patch << " -> making merge buf" << std::endl;

            u32 len_main = pdat_buf.element_count;

            {
                const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst = interface_hndl.get_interface_list(id_patch);
                for (auto & [int_pid, pdat_ptr] : p_interf_lst) {
                    len_main += (pdat_ptr->pos_s.size() + pdat_ptr->pos_d.size());
                }
            }

            using namespace patchdata_layout;

            merge_pdat_buf[id_patch].element_count = len_main;
            if(nVarpos_s > 0) merge_pdat_buf[id_patch].pos_s = std::make_unique<sycl::buffer<f32_3>>(nVarpos_s * len_main);
            if(nVarpos_d > 0) merge_pdat_buf[id_patch].pos_d = std::make_unique<sycl::buffer<f64_3>>(nVarpos_d * len_main);
            if(nVarU1_s  > 0) merge_pdat_buf[id_patch].U1_s  = std::make_unique<sycl::buffer<f32>>  (nVarU1_s  * len_main);
            if(nVarU1_d  > 0) merge_pdat_buf[id_patch].U1_d  = std::make_unique<sycl::buffer<f64>>  (nVarU1_d  * len_main);
            if(nVarU3_s  > 0) merge_pdat_buf[id_patch].U3_s  = std::make_unique<sycl::buffer<f32_3>>(nVarU3_s  * len_main);
            if(nVarU3_d  > 0) merge_pdat_buf[id_patch].U3_d  = std::make_unique<sycl::buffer<f64_3>>(nVarU3_d  * len_main);


            u32 offset_pos_s = 0;
            u32 offset_pos_d = 0;
            u32 offset_U1_s  = 0;
            u32 offset_U1_d  = 0;
            u32 offset_U3_s  = 0;
            u32 offset_U3_d  = 0;

            if(nVarpos_s > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.pos_s->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].pos_s->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_pos_s += pdat_buf.element_count*nVarpos_s;
            }

            if(nVarpos_d > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.pos_d->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].pos_d->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_pos_d += pdat_buf.element_count*nVarpos_d;
            }

            if(nVarU1_s > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.U1_s->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].U1_s->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_U1_s += pdat_buf.element_count*nVarU1_s;
            }

            if(nVarU1_d > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.U1_d->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].U1_d->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_U1_d += pdat_buf.element_count*nVarU1_d;
            }

            if(nVarU3_s > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.U3_s->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].U3_s->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU3_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_U3_s += pdat_buf.element_count*nVarU3_s;
            }

            if(nVarU3_d > 0){
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = pdat_buf.U3_d->get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_buf[id_patch].U3_d->get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU3_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
                });
                offset_U3_d += pdat_buf.element_count*nVarU3_d;
            }

            

            interface_hndl.for_each_interface(
                id_patch, 
                hndl.get_queue_compute(0), 
                [&](u64 patch_id, u64 interf_patch_id, PatchDataBuffer & interfpdat, std::tuple<f32_3,f32_3> box){

                    std::cout <<  "patch : n°"<< id_patch << " -> interface : "<<interf_patch_id << " merging" << std::endl;

                    min_box = sycl::min(std::get<0>(box),min_box);
                    max_box = sycl::min(std::get<1>(box),max_box);

                    if(nVarpos_s > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.pos_s->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].pos_s->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_pos_s;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarpos_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_pos_s += interfpdat.element_count*nVarpos_s;
                    }

                    if(nVarpos_d > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.pos_d->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].pos_d->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_pos_d;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarpos_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_pos_d += interfpdat.element_count*nVarpos_d;
                    }

                    if(nVarU1_s > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.U1_s->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].U1_s->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_U1_s;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU1_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_U1_s += interfpdat.element_count*nVarU1_s;
                    }

                    if(nVarU1_d > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.U1_d->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].U1_d->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_U1_d;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU1_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_U1_d += interfpdat.element_count*nVarU1_d;
                    }

                    if(nVarU3_s > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.U3_s->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].U3_s->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_U3_s;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU3_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_U3_s += interfpdat.element_count*nVarU3_s;
                    }

                    if(nVarU3_d > 0){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto source = interfpdat.U3_d->get_access<sycl::access::mode::read>(cgh);
                            auto dest = merge_pdat_buf[id_patch].U3_d->get_access<sycl::access::mode::discard_write>(cgh);
                            auto off = offset_U3_d;
                            cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU3_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                        });
                        offset_U3_d += interfpdat.element_count*nVarU3_d;
                    }
                }
            );

            merge_pdat_box[id_patch] = {min_box,max_box};
            

        });
        hndl.get_queue_compute(0).wait();
        tmerge_buf.stop();







        auto tgen_trees = timings::start_timer("radix tree compute", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, pos_vec>>> radix_trees;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
            std::cout << "patch : n°"<<id_patch << " -> making radix tree" << std::endl;

            PatchDataBuffer & mpdat_buf = merge_pdat_buf[id_patch];
            std::tuple<f32_3,f32_3> & box = merge_pdat_box[id_patch]; 

            //radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, pos_vec>>(hndl.get_queue_compute(0), box, mpdat_buf.get_pos<pos_vec>());
            
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;
            radix_trees[id_patch]->compute_cellvolume(hndl.get_queue_compute(0));
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;

            PatchDataBuffer & mpdat_buf = merge_pdat_buf[id_patch];

            radix_trees[id_patch]->template compute_int_boxes<
                DU1::nvar,
                DU1::ihpart
                >(hndl.get_queue_compute(0),mpdat_buf.get_U1<pos_prec>(),htol_up_tol);

        });
        hndl.get_queue_compute(0).wait();
        tgen_trees.stop();




        std::cout << "making omega field" << std::endl;
        PatchComputeField<f32> hnew_field;
        PatchComputeField<f32> omega_field;

        hnew_field.generate(sched);
        omega_field.generate(sched);

        hnew_field.to_sycl();
        omega_field.to_sycl();


        
        sched.for_each_patch([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
            std::cout << "patch : n°" << id_patch << "init h iter" << std::endl;
            
            sycl::buffer<f32> & hnew = *hnew_field.field_data_buf[id_patch];
            sycl::buffer<f32> eps_h = sycl::buffer<f32>(pdat_buf.element_count);

            sycl::range range_npart{pdat_buf.element_count};

            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                
                auto ha0 = pdat_buf.get_U1<f32>()->get_access<sycl::access::mode::read>(cgh);
                auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
                auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

                cgh.parallel_for<class Init_iterate_h>( range_npart, [=](sycl::item<1> item) {
                        
                    u32 id_a = (u32) item.get_id(0);

                    h[id_a] = ha0[id_a];
                    eps[id_a] = 100;

                });

            });



            /*
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                using u1_acc = decltype( pdat_buf.U1_s->get_access<sycl::access::mode::read>(cgh));
                using r_acc = decltype( pdat_buf.pos_s->get_access<sycl::access::mode::read>(cgh));

                u1_acc U1 = pdat_buf.U1_s->get_access<sycl::access::mode::read>(cgh);
                r_acc r = pdat_buf.pos_s->get_access<sycl::access::mode::read>(cgh);
                walker::Radix_tree_accessor<u32, f32_3> tree_acc(*radix_trees[id_patch], cgh);

                //std::vector<u1_acc> int_u1;

                //sched.for_each_patch(Function &&fct)

                



                auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                

                cgh.parallel_for<class SPHTest>(sycl::range(pdat_buf.pos_s->size()), [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);

                    f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                    f32 h_a = U1[id_a*DU1::nvar + DU1::ihpart];

                    f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                    f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                    f32_3 sum_axyz{0,0,0};

                    walker::rtree_for(
                        tree_acc,
                        [&](u32 node_id) {
                            f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                            float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            f32_3 dr = xyz_a - r[id_b];
                            f32 rab = sycl::length(dr);
                            f32 h_b = U1[id_b*DU1::nvar + DU1::ihpart];

                            if(rab > h_a*Kernel::Rkern && rab > h_b*Kernel::Rkern) return;

                            f32_3 r_ab_unit = dr / rab;

                            if(rab < 1e-9){
                                r_ab_unit = {0,0,0};
                            }

                            sum_axyz += f32_3{};

                        },
                        [](u32 node_id) {});
                });
            }); 
            */

            


        });

        hnew_field.to_map();
        omega_field.to_map();

        std::cout << "echange interface omega" << std::endl;
        PatchComputeFieldInterfaces<pos_prec> omega_field_interfaces = interface_hndl.template comm_interfaces_field<pos_prec>(sched,omega_field);






        

        
    }




};