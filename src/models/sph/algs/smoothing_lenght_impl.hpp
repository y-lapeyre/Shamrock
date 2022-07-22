// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/patchdata_buffer.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "models/sph/base/kernels.hpp"
#include "models/sph/base/sphpart.hpp"
#include "core/tree/radix_tree.hpp"

namespace impl {




    template<class flt>
    inline void sycl_init_h_iter_bufs(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchData & pdat_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h

        );

    template<>
    inline void sycl_init_h_iter_bufs<f32>(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchData & pdat_merge,
        sycl::buffer<f32> & hnew,
        sycl::buffer<f32> & omega,
        sycl::buffer<f32> & eps_h

        ){

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {
            
            auto acc_hpart = pdat_merge.fields_f32[ihpart].get_buf()->get_access<sycl::access::mode::read>(cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h_f32>( range_npart, [=](sycl::item<1> item) {
                    
                u32 id_a = (u32) item.get_id(0);

                h[id_a] = acc_hpart[id_a];
                eps[id_a] = 100;

            });

        });
    }






























    template<class flt>
    inline void sycl_init_h_iter_bufs(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h

        );

    template<>
    inline void sycl_init_h_iter_bufs<f32>(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<f32> & hnew,
        sycl::buffer<f32> & omega,
        sycl::buffer<f32> & eps_h

        ){

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {
            
            auto acc_hpart = pdat_buf_merge.fields_f32[ihpart]->get_access<sycl::access::mode::read>(cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h>( range_npart, [=](sycl::item<1> item) {
                    
                u32 id_a = (u32) item.get_id(0);

                h[id_a] = acc_hpart[id_a];
                eps[id_a] = 100;

            });

        });
    }



    template<class A,class B,class C> class Kernel_Iterh;
    template<class A,class B,class C> class Kernel_Finalize_omega;

    template<class morton_prec, class Kernel>
    class IntSmoothingLenghtCompute{public:








        template<class flt>
        static void sycl_h_iter_step(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchData & pdat_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void sycl_h_iter_step<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchData & pdat_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};
            
            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto eps = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                auto acc_hpart = pdat_merge.fields_f32.at(ihpart).get_buf()->get_access<sycl::access::mode::read>(cgh);
                auto r = pdat_merge.fields_f32_3[ixyz].get_buf()->get_access<sycl::access::mode::read>(cgh);
                
                
                Rta tree_acc(radix_t, cgh);



                auto cell_int_r = radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_iter;
                const f32 h_max_evol_m = 1/htol_up_iter;

                cgh.parallel_for<Kernel_Iterh<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);


                    if(eps[id_a] > 1e-6){

                        f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                        f32 h_a = h_new[id_a];
                        //f32 h_a2 = h_a*h_a;

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        f32 rho_sum = 0;
                        f32 sumdWdh = 0;
                        
                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                            cur_pos_max_cell_b, int_r_max_cell);
                            },
                            [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&sumdWdh](u32 id_b) {
                                //f32_3 dr = xyz_a - r[id_b];
                                f32 rab = sycl::distance( xyz_a , r[id_b]);

                                if(rab > h_a*Kernel::Rkern) return;

                                //f32 rab = sycl::sqrt(rab2);

                                rho_sum += part_mass*Kernel::W(rab,h_a);
                                sumdWdh += part_mass*Kernel::dhW(rab,h_a);

                            },
                            [](u32 node_id) {});
                        

                        
                        f32 rho_ha = rho_h(part_mass, h_a);

                        f32 f_iter = rho_sum - rho_ha;
                        f32 df_iter = sumdWdh + 3*rho_ha/h_a;

                        //f32 omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
                        //f32 new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

                        f32 new_h = h_a - f_iter/df_iter;


                        if(new_h < h_a*h_max_evol_m) new_h = h_max_evol_m*h_a;
                        if(new_h > h_a*h_max_evol_p) new_h = h_max_evol_p*h_a;

                        
                        f32 ha_0 = acc_hpart[id_a];
                        
                        
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



        }






        template<class flt>
        static void sycl_h_iter_omega(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchData & pdat_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void sycl_h_iter_omega<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchData & pdat_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){
            
            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};


            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_merge.fields_f32_3.at(ixyz).get_buf()->get_access<sycl::access::mode::read>(cgh);
                
                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(radix_t, cgh);



                auto cell_int_r =radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_tol;
                const f32 h_max_evol_m = 1/htol_up_tol;

                cgh.parallel_for<Kernel_Finalize_omega<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);

                    f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                    f32 h_a = h_new[id_a];
                    //f32 h_a2 = h_a*h_a;

                    f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                    f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                    f32 rho_sum = 0;
                    f32 part_omega_sum = 0;
                    
                    walker::rtree_for(
                        tree_acc,
                        [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                            f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                            float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                        },
                        [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&part_omega_sum](u32 id_b) {
                            //f32_3 dr = xyz_a - r[id_b];
                            f32 rab = sycl::distance( xyz_a , r[id_b]);

                            if(rab > h_a*Kernel::Rkern) return;

                            //f32 rab = sycl::sqrt(rab2);

                            rho_sum += part_mass*Kernel::W(rab,h_a);
                            part_omega_sum += part_mass * Kernel::dhW(rab,h_a);

                        },
                        [](u32 node_id) {});
                    

                    
                    f32 rho_ha = rho_h(part_mass, h_a);
                    omga[id_a] = 1 + (h_a/(3*rho_ha))*part_omega_sum;
                    

                });

            }); 
        }






































        

        template<class flt>
        static void sycl_h_iter_step(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void sycl_h_iter_step<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};
            
            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto eps = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                auto acc_hpart = pdat_buf_merge.fields_f32.at(ihpart)->get_access<sycl::access::mode::read>(cgh);
                auto r = pdat_buf_merge.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);
                
                
                Rta tree_acc(radix_t, cgh);



                auto cell_int_r = radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_iter;
                const f32 h_max_evol_m = 1/htol_up_iter;

                cgh.parallel_for<Kernel_Iterh<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);


                    if(eps[id_a] > 1e-6){

                        f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                        f32 h_a = h_new[id_a];
                        //f32 h_a2 = h_a*h_a;

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        f32 rho_sum = 0;
                        f32 sumdWdh = 0;
                        
                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                            cur_pos_max_cell_b, int_r_max_cell);
                            },
                            [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&sumdWdh](u32 id_b) {
                                //f32_3 dr = xyz_a - r[id_b];
                                f32 rab = sycl::distance( xyz_a , r[id_b]);

                                if(rab > h_a*Kernel::Rkern) return;

                                //f32 rab = sycl::sqrt(rab2);

                                rho_sum += part_mass*Kernel::W(rab,h_a);
                                sumdWdh += part_mass*Kernel::dhW(rab,h_a);

                            },
                            [](u32 node_id) {});
                        

                        
                        f32 rho_ha = rho_h(part_mass, h_a);

                        f32 f_iter = rho_sum - rho_ha;
                        f32 df_iter = sumdWdh + 3*rho_ha/h_a;

                        //f32 omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
                        //f32 new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

                        f32 new_h = h_a - f_iter/df_iter;


                        if(new_h < h_a*h_max_evol_m) new_h = h_max_evol_m*h_a;
                        if(new_h > h_a*h_max_evol_p) new_h = h_max_evol_p*h_a;

                        
                        f32 ha_0 = acc_hpart[id_a];
                        
                        
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
        }












        template<class flt>
        static void sycl_h_iter_omega(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void sycl_h_iter_omega<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){
            
            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};


            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_buf_merge.fields_f32_3.at(ixyz)->get_access<sycl::access::mode::read>(cgh);
                
                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(radix_t, cgh);



                auto cell_int_r =radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_tol;
                const f32 h_max_evol_m = 1/htol_up_tol;

                cgh.parallel_for<Kernel_Finalize_omega<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);

                    f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                    f32 h_a = h_new[id_a];
                    //f32 h_a2 = h_a*h_a;

                    f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                    f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                    f32 rho_sum = 0;
                    f32 part_omega_sum = 0;
                    
                    walker::rtree_for(
                        tree_acc,
                        [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                            f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                            float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                        },
                        [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&part_omega_sum](u32 id_b) {
                            //f32_3 dr = xyz_a - r[id_b];
                            f32 rab = sycl::distance( xyz_a , r[id_b]);

                            if(rab > h_a*Kernel::Rkern) return;

                            //f32 rab = sycl::sqrt(rab2);

                            rho_sum += part_mass*Kernel::W(rab,h_a);
                            part_omega_sum += part_mass * Kernel::dhW(rab,h_a);

                        },
                        [](u32 node_id) {});
                    

                    
                    f32 rho_ha = rho_h(part_mass, h_a);
                    omga[id_a] = 1 + (h_a/(3*rho_ha))*part_omega_sum;
                    

                });

            }); 
        }


    };


    

}