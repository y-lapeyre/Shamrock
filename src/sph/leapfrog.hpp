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
#include "forces.hpp"
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

constexpr f32 gpart_mass = 2e-4;

template <class flt,class Kernel,class u_morton> class SPHTimestepperLeapfrogAlgs {public:

    using vec3 = sycl::vec<flt, 3>;

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value || std::is_same<flt, f64>::value
    , "Leapfrog : floating point type should be one of (f16,f32,f64)");

    inline static void sycl_leapfrog_predictor(sycl::queue &queue, u32 npart, flt dt, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_vxyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_axyz) {

        sycl::range<1> range_npart{npart};

        auto ker_predict_step = [&](sycl::handler &cgh) {
            auto acc_xyz  = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_vxyz = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id();

                vec3 &vxyz = acc_vxyz[item];
                vec3 &axyz = acc_axyz[item];

                // v^{n + 1/2} = v^n + dt/2 a^n
                vxyz = vxyz + (dt / 2) * (axyz);

                // r^{n + 1} = r^n + dt v^{n + 1/2}
                acc_xyz[item] = acc_xyz[item] + dt * vxyz;

                // v^* = v^{n + 1/2} + dt/2 a^n
                vxyz = vxyz + (dt / 2) * (axyz);
            });
        };

        queue.submit(ker_predict_step);
    }

    inline static void sycl_leapfrog_corrector(sycl::queue &queue, u32 npart, flt dt, std::unique_ptr<sycl::buffer<vec3>> &buf_vxyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_axyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_axyz_old) {

        sycl::range<1> range_npart{npart};

        auto ker_corect_step = [&](sycl::handler &cgh) {
            auto acc_vxyz     = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz     = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = buf_axyz_old->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id();

                vec3 &vxyz     = acc_vxyz[item];
                vec3 &axyz     = acc_axyz[item];
                vec3 &axyz_old = acc_axyz_old[item];

                // v^* = v^{n + 1/2} + dt/2 a^n
                vxyz = vxyz + (dt / 2) * (axyz - axyz_old);
            });
        };

        queue.submit(ker_corect_step);
    }

    inline static void sycl_swap_a_field(sycl::queue &queue, u32 npart, std::unique_ptr<sycl::buffer<vec3>> &buf_axyz,
                                std::unique_ptr<sycl::buffer<vec3>> &buf_axyz_old) {
        sycl::range<1> range_npart{npart};

        auto ker_swap_a = [&](sycl::handler &cgh) {
            auto acc_axyz     = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = buf_axyz_old->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                vec3 axyz     = acc_axyz[item];
                vec3 axyz_old = acc_axyz_old[item];

                acc_axyz[item]     = axyz_old;
                acc_axyz_old[item] = axyz;
            });
        };

        queue.submit(ker_swap_a);
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
    





    SPHTimestepperLeapfrogAlgs(SchedulerMPI &sched,bool periodic_mode,flt htol_up_tol,
        flt htol_up_iter ) : sched(sched), periodic_mode(periodic_mode) , htol_up_tol(htol_up_tol) , htol_up_iter(htol_up_iter){}



    inline flt step(flt old_time, bool do_force, bool do_corrector){


        const flt loc_htol_up_tol  = htol_up_tol;
        const flt loc_htol_up_iter = htol_up_iter;







        SyCLHandler &hndl = SyCLHandler::get_instance();

        SerialPatchTree<vec3> sptree(sched.patch_tree, sched.get_box_tranform<vec3>());
        sptree.attach_buf();

        const u32 ixyz      = sched.pdl.get_field_idx<f32_3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<f32_3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<f32_3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<f32_3>("axyz_old");

        const u32 ihpart = sched.pdl.get_field_idx<f32>("hpart");

        GlobalVariable<min,f32> cfl_glb_var;

        cfl_glb_var.compute_var_patch(sched, [&](u64 id_patch, PatchDataBuffer &pdat_buf) {
            u32 npart_patch = pdat_buf.element_count;

            std::unique_ptr<sycl::buffer<f32>> buf_cfl = std::make_unique<sycl::buffer<f32>>(npart_patch);

            sycl::range<1> range_npart{npart_patch};

            auto ker_Reduc_step_mincfl = [&](sycl::handler &cgh) {
                auto arr = buf_cfl->get_access<sycl::access::mode::discard_write>(cgh);

                auto acc_hpart = pdat_buf.fields_f32[ihpart]->template get_access<sycl::access::mode::read>(cgh);
                auto acc_axyz  = pdat_buf.fields_f32_3[iaxyz]->template get_access<sycl::access::mode::read>(cgh);

                f32 cs = 1;

                constexpr f32 C_cour  = 0.1;
                constexpr f32 C_force = 0.1;

                cgh.parallel_for<class Initial_dtcfl>(range_npart, [=](sycl::item<1> item) {
                    u32 i = (u32)item.get_id(0);

                    f32 h_a    = acc_hpart[item];
                    f32_3 axyz = acc_axyz[item];

                    f32 dtcfl_P = C_cour * h_a / cs;
                    f32 dtcfl_a = C_force * sycl::sqrt(h_a / sycl::length(axyz));

                    arr[i] = sycl::min(dtcfl_P, dtcfl_a);
                });
            };

            hndl.get_queue_compute(0).submit(ker_Reduc_step_mincfl);

            f32 min_cfl = syclalg::get_min<f32>(hndl.get_queue_compute(0), buf_cfl);





            return min_cfl;
        });

        cfl_glb_var.reduce_val();


        f32 cfl_val = cfl_glb_var.get_val();

        f32 dt_cur = sycl::min(f32(0.001),cfl_val);

        std::cout << " --- current dt  : " << dt_cur << std::endl;

        flt step_time = old_time;
        step_time += dt_cur;

        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer &pdat_buf) {
            std::cout << "patch : n°" << id_patch << " -> leapfrog predictor" << std::endl;

            SPHTimestepperLeapfrogAlgs<flt,Kernel,u_morton>::sycl_leapfrog_predictor(hndl.get_queue_compute(0), pdat_buf.element_count, dt_cur,
                                              pdat_buf.fields_f32_3.at(ixyz), pdat_buf.fields_f32_3.at(ivxyz),
                                              pdat_buf.fields_f32_3.at(iaxyz));

            std::cout << "patch : n°" << id_patch << " -> a field swap" << std::endl;

            SPHTimestepperLeapfrogAlgs<flt,Kernel,u_morton>::sycl_swap_a_field(hndl.get_queue_compute(0), pdat_buf.element_count, pdat_buf.fields_f32_3.at(iaxyz),
                                        pdat_buf.fields_f32_3.at(iaxyz_old));

            if (periodic_mode) {
                SPHTimestepperLeapfrogAlgs<flt,Kernel,u_morton>::sycl_position_modulo(hndl.get_queue_compute(0), pdat_buf.element_count,
                                               pdat_buf.fields_f32_3[ixyz], sched.get_box_volume<vec3>());
            }
        });

        std::cout << "particle reatribution" << std::endl;
        reatribute_particles(sched, sptree, periodic_mode);

        

        std::cout << "exhanging interfaces" << std::endl;
        auto timer_h_max = timings::start_timer("compute_hmax", timings::timingtype::function);

        std::cout << "owned_data : " << std::endl;
        for (auto &[k, a] : sched.patch_data.owned_data) {
            std::cout << " pdat : " << k << std::endl;
        }

        PatchField<f32> h_field;
        sched.compute_patch_field(
            h_field, get_mpi_type<flt>(), [loc_htol_up_tol](sycl::queue &queue, Patch &p, PatchDataBuffer &pdat_buf) {
                return patchdata::sph::get_h_max<flt>(pdat_buf.pdl, queue, pdat_buf) * loc_htol_up_tol * Kernel::Rkern;
            });

        timer_h_max.stop();

        InterfaceHandler<vec3, flt> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<vec3, flt>>(sched, sptree, h_field,
                                                                                                 periodic_mode);
        interface_hndl.comm_interfaces(sched, periodic_mode);

        // merging strat
        auto tmerge_buf = timings::start_timer("buffer merging", timings::sycl);
        std::unordered_map<u64, MergedPatchDataBuffer<vec3>> merge_pdat_buf;
        make_merge_patches(sched, interface_hndl, merge_pdat_buf);
        hndl.get_queue_compute(0).wait();
        tmerge_buf.stop();


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

        std::cout << "making omega field" << std::endl;
        PatchComputeField<f32> hnew_field;
        PatchComputeField<f32> omega_field;

        hnew_field.generate(sched);
        omega_field.generate(sched);

        hnew_field.to_sycl();
        omega_field.to_sycl();

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

            sph::algs::SmoothingLenghtCompute<f32, u32, Kernel> h_iterator(sched.pdl, htol_up_tol, htol_up_iter);

            h_iterator.iterate_smoothing_lenght(hndl.get_queue_compute(0), merge_pdat_buf.at(id_patch).or_element_cnt,
                                                gpart_mass, *radix_trees[id_patch], pdat_buf_merge, hnew, omega, eps_h);

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

        //

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

        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> hnew_field_merged;
        make_merge_patches_comp_field<f32>(sched, interface_hndl, hnew_field, hnew_field_interfaces, hnew_field_merged);
        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> omega_field_merged;
        make_merge_patches_comp_field<f32>(sched, interface_hndl, omega_field, omega_field_interfaces, omega_field_merged);



        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            if (merge_pdat_buf.at(id_patch).or_element_cnt == 0)
                std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

            sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf;
            sycl::buffer<f32> &omega = *omega_field_merged[id_patch].buf;

            sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

            if (do_force) {
                std::cout << "patch : n°" << id_patch << "compute forces" << std::endl;
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                    auto omga  = omega.get_access<sycl::access::mode::read>(cgh);

                    auto r        = pdat_buf_merge.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);
                    auto acc_axyz = pdat_buf_merge.fields_f32_3[iaxyz]->get_access<sycl::access::mode::discard_write>(cgh);

                    using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                    Rta tree_acc(*radix_trees[id_patch], cgh);

                    auto cell_int_r =
                        radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                    f32 part_mass = gpart_mass;
                    f32 cs        = 1;

                    const f32 htol = htol_up_tol;

                    // sycl::stream out(65000,65000,cgh);

                    cgh.parallel_for<class forces>(range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32)item.get_id(0);

                        f32_3 sum_axyz = {0, 0, 0};
                        f32 h_a        = h_new[id_a];

                        f32_3 xyz_a = r[id_a];

                        f32 rho_a    = rho_h(part_mass, h_a);
                        f32 rho_a_sq = rho_a * rho_a;

                        f32 P_a     = cs * cs * rho_a;
                        f32 omega_a = omga[id_a];

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc, &xyz_a, &inter_box_a_min, &inter_box_a_max, &cell_int_r,&htol](u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern * htol;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                           cur_pos_max_cell_b, int_r_max_cell);
                            },
                            [&](u32 id_b) {
                                // compute only omega_a
                                f32_3 dr = xyz_a - r[id_b];
                                f32 rab  = sycl::length(dr);
                                f32 h_b  = h_new[id_b];

                                if (rab > h_a * Kernel::Rkern && rab > h_b * Kernel::Rkern)
                                    return;

                                f32_3 r_ab_unit = dr / rab;

                                if (rab < 1e-9) {
                                    r_ab_unit = {0, 0, 0};
                                }

                                f32 rho_b   = rho_h(part_mass, h_b);
                                f32 P_b     = cs * cs * rho_b;
                                f32 omega_b = omga[id_b];

                                f32_3 tmp = sph_pressure<f32_3, f32>(part_mass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
                                                                     omega_b, 0, 0, r_ab_unit * Kernel::dW(rab, h_a),
                                                                     r_ab_unit * Kernel::dW(rab, h_b));

                                sum_axyz += tmp;
                            },
                            [](u32 node_id) {});

                        // out << "sum : " << sum_axyz << "\n";

                        acc_axyz[id_a] = sum_axyz;
                    });
                });
            }

            if (do_corrector) {

                std::cout << "leapfrog corrector " << std::endl;

                SPHTimestepperLeapfrogAlgs<flt,Kernel,u_morton>::sycl_leapfrog_corrector(hndl.get_queue_compute(0), merge_pdat_buf.at(id_patch).or_element_cnt,
                                                  dt_cur, pdat_buf_merge.fields_f32_3[ivxyz],
                                                  pdat_buf_merge.fields_f32_3[iaxyz],
                                                  pdat_buf_merge.fields_f32_3[iaxyz_old]);
            }
        });

        write_back_merge_patches(sched, interface_hndl, merge_pdat_buf);

        return step_time;
    }

};

template <class flt> class SPHTimestepperLeapfrog {
  public:
    using pos_vec = sycl::vec<flt, 3>;

    using u_morton = u32;

    using Kernel = sph::kernels::M4<f32>;

    inline void step(SchedulerMPI &sched, std::string dump_folder, u32 step_cnt, f64 &step_time) {

        bool periodic_bc = true;

        flt htol_up_tol  = 1.4;
        flt htol_up_iter = 1.2;

        SPHTimestepperLeapfrogAlgs<flt, Kernel, u_morton> stepper(sched,periodic_bc,htol_up_tol,htol_up_iter);

        bool do_force = step_cnt > 4;
        bool do_corrector = step_cnt > 5;

        step_time = stepper.step(step_time, do_force, do_corrector);
    }
};
