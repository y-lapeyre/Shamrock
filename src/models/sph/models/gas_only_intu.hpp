#pragma once

#include "aliases.hpp"
#include "models/generic/algs/cfl_utils.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "models/sph/forces.hpp"
#include "models/sph/integrators/leapfrog.hpp"
#include "models/generic/algs/integrators_utils.hpp"


namespace models::sph {

template<class flt, class u_morton, class Kernel>
class GasOnlyInternalU{public: 
    flt gpart_mass;
    bool periodic_bc = true;

    flt htol_up_tol  = 1.4;
    flt htol_up_iter = 1.2;

    flt cfl_cour  = 0.1;
    flt cfl_force = 0.1;

    bool do_force ;
    bool do_corrector ;

    const flt eos_cs = 1; //TODO move to EOS

    using vec3 = sycl::vec<flt, 3>;
    using Stepper = integrators::sph::LeapfrogGeneral<flt, Kernel, u_morton>;

    const flt gamma_eos = 5./3.;


    void step(PatchScheduler &sched, f64 &step_time) {

        Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);


        SyCLHandler &hndl = SyCLHandler::get_instance();

        const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

        const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

        const u32 iu    = sched.pdl.get_field_idx<flt>("u");
        const u32 idu    = sched.pdl.get_field_idx<flt>("du");
        const u32 idu_old    = sched.pdl.get_field_idx<flt>("du_old");

        PatchComputeField<f32> pressure_field;


        step_time = stepper.step(step_time, do_force, do_corrector, 
        
        [&](u64 id_patch, PatchDataBuffer &pdat_buf) {

            
                flt cfl_val = CflUtility<flt>::basic_cfl(pdat_buf, [&](sycl::handler &cgh, sycl::buffer<flt> & buf_cfl, sycl::range<1> range_it){

                    auto arr = buf_cfl.template get_access<sycl::access::mode::discard_write>(cgh);
                    auto acc_hpart = pdat_buf.get_field<flt>(ihpart)->template get_access<sycl::access::mode::read>(cgh);
                    auto acc_axyz  = pdat_buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read>(cgh);

                    const flt cs = eos_cs;

                    const flt c_cour  = cfl_cour;
                    const flt c_force = cfl_force;

                    cgh.parallel_for<class Initial_dtcfl>(range_it, [=](sycl::item<1> item) {
                        u32 i = (u32)item.get_id(0);

                        flt h_a    = acc_hpart[item];
                        vec3 axyz = acc_axyz[item];

                        flt dtcfl_p = c_cour * h_a / cs;
                        flt dtcfl_a = c_force * sycl::sqrt(h_a / sycl::length(axyz));

                        arr[i] = sycl::min(dtcfl_p, dtcfl_a);
                    });

                });

                std::cout << "cfl dt : " << cfl_val << std::endl;

                return sycl::min(f32(0.001),cfl_val);

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ,flt hdt){
                
                sycl::buffer<vec3> & vxyz =  * buf.get_field<vec3>(ivxyz);
                sycl::buffer<vec3> & axyz =  * buf.get_field<vec3>(iaxyz);

                field_advance_time(queue, vxyz, axyz, range_npart, hdt);

                sycl::buffer<flt> & u =  * buf.get_field<flt>(iu);
                sycl::buffer<flt> & du =  * buf.get_field<flt>(idu);

                field_advance_time(queue, u, du, range_npart, hdt);

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ){

                queue.submit([&](sycl::handler &cgh) {
                    auto acc_axyz = buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old)->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        vec3 axyz     = acc_axyz[item];
                        vec3 axyz_old = acc_axyz_old[item];

                        acc_axyz[item]     = axyz_old;
                        acc_axyz_old[item] = axyz;

                    });
                });


                queue.submit([&](sycl::handler &cgh) {
                    auto acc_du = buf.get_field<flt>(idu)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_du_old = buf.get_field<flt>(idu_old)->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        flt du     = acc_du[item];
                        flt du_old = acc_du_old[item];

                        acc_du[item]     = du_old;
                        acc_du_old[item] = du;

                    });
                });


            },
            [&](PatchScheduler & sched, 
                std::unordered_map<u64, MergedPatchDataBuffer<vec3>>& merge_pdat_buf, 
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& hnew_field_merged,
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& omega_field_merged
                ) {


                    std::unordered_map<u64, u32> size_map;

                    for(auto & [k,buf] : merge_pdat_buf){
                        size_map[k] = buf.data->element_count;
                    }

                    pressure_field.generate(sched,size_map);
                    pressure_field.to_sycl();

                    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                        sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf;
                        sycl::buffer<f32> &press  = *pressure_field.field_data_buf[id_patch];

                        PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

                        sycl::range range_npart{hnew.size()};

                        auto cs = eos_cs;
                        auto part_mass = gpart_mass;

                        const flt gamma = gamma_eos;

                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto h = hnew.get_access<sycl::access::mode::read>(cgh);

                            auto acc_u = pdat_buf_merge.get_field<flt>(iu)->template get_access<sycl::access::mode::read>(cgh);

                            auto p = press.get_access<sycl::access::mode::discard_write>(cgh);

                            cgh.parallel_for(range_npart,
                                    [=](sycl::item<1> item) { 
                                        
                                        p[item] = (gamma-1) * rho_h(part_mass, h[item]) *acc_u[item] ; 
                                        
                                        
                                        });
                        });

                    });
                }
            ,
            [&](


                PatchScheduler & sched, 
                std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, vec3>>>& radix_trees,
                std::unordered_map<u64, MergedPatchDataBuffer<vec3>>& merge_pdat_buf, 
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& hnew_field_merged,
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& omega_field_merged,
                flt htol_up_tol
                ){
                sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                    if (merge_pdat_buf.at(id_patch).or_element_cnt == 0){
                        std::cout << " empty => skipping" << std::endl;return;
                    }

                    SyCLHandler &hndl = SyCLHandler::get_instance();

                    PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

                    sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf;
                    sycl::buffer<f32> &omega = *omega_field_merged[id_patch].buf;

                    sycl::buffer<f32> &press  = *pressure_field.field_data_buf[id_patch];

                    sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

                
                    std::cout << "patch : n°" << id_patch << "compute forces" << std::endl;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                        auto omga  = omega.get_access<sycl::access::mode::read>(cgh);

                        auto r        = pdat_buf_merge.get_field<f32_3>(ixyz)->get_access<sycl::access::mode::read>(cgh);

                        auto vxyz        = pdat_buf_merge.get_field<f32_3>(ivxyz)->get_access<sycl::access::mode::read>(cgh);

                        auto acc_axyz = pdat_buf_merge.get_field<f32_3>(iaxyz)->get_access<sycl::access::mode::discard_write>(cgh);

                        auto acc_du = pdat_buf_merge.get_field<flt>(idu)->template get_access<sycl::access::mode::discard_write>(cgh);

                        using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                        Rta tree_acc(*radix_trees[id_patch], cgh);

                        auto cell_int_r =
                            radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);


                        auto pres = press.get_access<sycl::access::mode::read>(cgh);

                        const f32 part_mass = gpart_mass;
                        //const f32 cs        = eos_cs;

                        const f32 htol = htol_up_tol;

                        const flt gamma = gamma_eos;

                        //sycl::stream out(1024,1024,cgh);

                        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                            u32 id_a = (u32)item.get_id(0);

                            f32_3 sum_axyz = {0, 0, 0};
                            f32 h_a        = h_new[id_a];

                            f32_3 xyz_a = r[id_a];

                            f32 rho_a    = rho_h(part_mass, h_a);
                            f32 rho_a_sq = rho_a * rho_a;

                            f32 P_a     = pres[id_a];
                            f32_3 v_a = vxyz[id_a];
                            f32 cs_a = sycl::sqrt(gamma*P_a/rho_a);
                            //f32 P_a     = cs * cs * rho_a;
                            f32 omega_a = omga[id_a];

                            f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            f32 psum_u = 0;

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

                                    f32_3 v_b = vxyz[id_b];

                                    f32_3 v_ab = v_a - v_b;

                                    f32 v_ab_rabu = sycl::dot(v_ab,r_ab_unit);

                                    const f32 alpha_av = 1;
                                    const f32 beta_av = 2;
                                    

                                    f32 rho_b   = rho_h(part_mass, h_b);
                                    f32 P_b     = pres[id_b];
                                    //f32 P_b     = cs * cs * rho_b;
                                    f32 omega_b = omga[id_b];

                                    f32 cs_b = sycl::sqrt(gamma*P_b/rho_b);

                                    f32 vsig_a = alpha_av*cs_a + beta_av*sycl::abs(v_ab_rabu);
                                    f32 vsig_b = alpha_av*cs_b + beta_av*sycl::abs(v_ab_rabu);

                                    f32 qa_ab = 0;
                                    f32 qb_ab = 0;

                                    if (v_ab_rabu < 0){
                                        qa_ab = -f32(0.5f)*rho_a*vsig_a*v_ab_rabu;
                                    }

                                    if (v_ab_rabu < 0){
                                        qb_ab = -f32(0.5f)*rho_b*vsig_b*v_ab_rabu;
                                    }

                                    f32_3 tmp = sph_pressure<f32_3, f32>(part_mass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
                                                                        omega_b, qa_ab, qb_ab, r_ab_unit * Kernel::dW(rab, h_a),
                                                                        r_ab_unit * Kernel::dW(rab, h_b));

                                    sum_axyz += tmp;

                                    f32 tmp_du = part_mass*sycl::dot(v_ab,Kernel::dW(rab, h_a) * r_ab_unit);


                                    psum_u += tmp_du;
                                },
                                [](u32 node_id) {});

                            // out << "sum : " << sum_axyz << "\n";

                            acc_axyz[id_a] = sum_axyz;


                            
                            f32 du = psum_u*P_a/(rho_a_sq*omega_a);
                            
                           

                            acc_du[id_a] = du;
                        });
                    });
                    

                });

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ,flt hdt){
                

                queue.submit([&](sycl::handler &cgh) {
                    auto acc_vxyz     = buf.get_field<vec3>(ivxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz     = buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old)->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 gid = (u32)item.get_id();

                        vec3 &vxyz     = acc_vxyz[item];
                        vec3 &axyz     = acc_axyz[item];
                        vec3 &axyz_old = acc_axyz_old[item];

                        // v^* = v^{n + 1/2} + dt/2 a^n
                        vxyz = vxyz + (hdt) * (axyz - axyz_old);
                    });
                });


                queue.submit([&](sycl::handler &cgh) {
                    auto acc_u     = buf.get_field<flt>(iu)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_du     = buf.get_field<flt>(idu)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_du_old = buf.get_field<flt>(idu_old)->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 gid = (u32)item.get_id();

                        flt &u     = acc_u[item];
                        flt &du     = acc_du[item];
                        flt &du_old = acc_du_old[item];

                        // v^* = v^{n + 1/2} + dt/2 a^n
                        u = u + (hdt) * (du - du_old);
                    });
                });
            });

    }

};

} // namespace models::sph