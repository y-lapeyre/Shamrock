#pragma once

#include "CL/sycl/buffer.hpp"
#include "CL/sycl/builtins.hpp"
#include "algs/syclreduction.hpp"
#include "aliases.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "io/logs.hpp"
#include "particles/particle_patch_mover.hpp"
#include "patch/compute_field.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/kernels.hpp"
#include "sph/sphpart.hpp"
#include "sph/sphpatch.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "tree/radix_tree.hpp"
#include <memory>
#include <mpi.h>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "forces.hpp"


constexpr f32 gpart_mass = 1e-5;


template<class vec>
struct MergedPatchDataBuffer {public:
    u32 or_element_cnt;
    PatchDataBuffer data;
    std::tuple<vec,vec> box;
};

template<class pos_prec,class pos_vec>
inline void make_merge_patches(
    SchedulerMPI & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,
    
    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){



    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        auto tmp_box = sched.patch_data.sim_box.get_box<pos_prec>(cur_p);

        f32_3 min_box = std::get<0>(tmp_box);
        f32_3 max_box = std::get<1>(tmp_box);

        std::cout << "patch : n°"<<id_patch << " -> making merge buf" << std::endl;

        u32 len_main = pdat_buf.element_count;
        merge_pdat_buf[id_patch].or_element_cnt = len_main;

        {
            const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst = interface_hndl.get_interface_list(id_patch);
            for (auto & [int_pid, pdat_ptr] : p_interf_lst) {
                len_main += (pdat_ptr->pos_s.size() + pdat_ptr->pos_d.size());
            }
        }

        using namespace patchdata_layout;

        merge_pdat_buf[id_patch].data.element_count = len_main;
        if(nVarpos_s > 0) merge_pdat_buf[id_patch].data.pos_s = std::make_unique<sycl::buffer<f32_3>>(nVarpos_s * len_main);
        if(nVarpos_d > 0) merge_pdat_buf[id_patch].data.pos_d = std::make_unique<sycl::buffer<f64_3>>(nVarpos_d * len_main);
        if(nVarU1_s  > 0) merge_pdat_buf[id_patch].data.U1_s  = std::make_unique<sycl::buffer<f32>>  (nVarU1_s  * len_main);
        if(nVarU1_d  > 0) merge_pdat_buf[id_patch].data.U1_d  = std::make_unique<sycl::buffer<f64>>  (nVarU1_d  * len_main);
        if(nVarU3_s  > 0) merge_pdat_buf[id_patch].data.U3_s  = std::make_unique<sycl::buffer<f32_3>>(nVarU3_s  * len_main);
        if(nVarU3_d  > 0) merge_pdat_buf[id_patch].data.U3_d  = std::make_unique<sycl::buffer<f64_3>>(nVarU3_d  * len_main);


        u32 offset_pos_s = 0;
        u32 offset_pos_d = 0;
        u32 offset_U1_s  = 0;
        u32 offset_U1_d  = 0;
        u32 offset_U3_s  = 0;
        u32 offset_U3_d  = 0;

        if(nVarpos_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.pos_s->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.pos_s->template get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            offset_pos_s += pdat_buf.element_count*nVarpos_s;
        }

        if(nVarpos_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.pos_d->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.pos_d->template get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            offset_pos_d += pdat_buf.element_count*nVarpos_d;
        }

        if(nVarU1_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.U1_s->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.U1_s->template get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            offset_U1_s += pdat_buf.element_count*nVarU1_s;
        }

        if(nVarU1_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.U1_d->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.U1_d->template get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            offset_U1_d += pdat_buf.element_count*nVarU1_d;
        }

        if(nVarU3_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.U3_s->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.U3_s->template get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU3_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            offset_U3_s += pdat_buf.element_count*nVarU3_s;
        }

        if(nVarU3_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.U3_d->get_access<sycl::access::mode::read>(cgh);
                auto dest = merge_pdat_buf[id_patch].data.U3_d->template get_access<sycl::access::mode::discard_write>(cgh);
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
                max_box = sycl::max(std::get<1>(box),max_box);

                if(nVarpos_s > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.pos_s->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.pos_s->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_pos_s;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarpos_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_pos_s += interfpdat.element_count*nVarpos_s;
                }

                if(nVarpos_d > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.pos_d->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.pos_d->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_pos_d;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarpos_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_pos_d += interfpdat.element_count*nVarpos_d;
                }

                if(nVarU1_s > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.U1_s->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.U1_s->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_U1_s;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU1_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_U1_s += interfpdat.element_count*nVarU1_s;
                }

                if(nVarU1_d > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.U1_d->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.U1_d->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_U1_d;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU1_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_U1_d += interfpdat.element_count*nVarU1_d;
                }

                if(nVarU3_s > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.U3_s->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.U3_s->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_U3_s;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU3_s}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_U3_s += interfpdat.element_count*nVarU3_s;
                }

                if(nVarU3_d > 0){
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.U3_d->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merge_pdat_buf[id_patch].data.U3_d->template get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = offset_U3_d;
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nVarU3_d}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    offset_U3_d += interfpdat.element_count*nVarU3_d;
                }
            }
        );

        merge_pdat_buf[id_patch].box = {min_box,max_box};


        
        

    });


}


template<class pos_prec,class pos_vec>
inline void write_back_merge_patches(
    SchedulerMPI & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,
    
    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){


    SyCLHandler &hndl = SyCLHandler::get_instance();

    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
        if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;


        std::cout << "patch : n°"<<id_patch << " -> write back merge buf" << std::endl;


        using namespace patchdata_layout;


        if(nVarpos_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.pos_s->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.pos_s->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        if(nVarpos_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.pos_d->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.pos_d->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarpos_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        if(nVarU1_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.U1_s->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.U1_s->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        if(nVarU1_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.U1_d->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.U1_d->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU1_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        if(nVarU3_s > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.U3_s->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.U3_s->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU3_s}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        if(nVarU3_d > 0){
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.U3_d->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf[id_patch].data.U3_d->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nVarU3_d}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

    });

}


template<class T>
class IntMergedPatchComputeField{public:
    

    u32 or_element_cnt;
    u32 tot_element_cnt;
    std::unique_ptr<sycl::buffer<T>> buf;

    inline void gen(
        std::vector<T> & patch_comp_field ,
        std::vector<std::tuple<u64, std::unique_ptr<std::vector<T>>>> &interfaces){


        SyCLHandler &hndl = SyCLHandler::get_instance();
        
        u32 len_main = patch_comp_field.size();
        or_element_cnt = len_main;

        {
            for (auto & [int_pid, pdat_ptr] : interfaces) {
                len_main += (pdat_ptr->size());
            }
        }

        tot_element_cnt = len_main;

        buf = std::make_unique<sycl::buffer<T>>(tot_element_cnt);

        sycl::buffer<T> buf_pcfield(patch_comp_field.data(),patch_comp_field.size());

        u32 offset = 0;

        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
            auto source = buf_pcfield.template get_access<sycl::access::mode::read>(cgh);
            auto dest = buf->template get_access<sycl::access::mode::discard_write>(cgh);
            cgh.parallel_for( sycl::range{or_element_cnt}, [=](sycl::item<1> item) { dest[item] = source[item]; });
        });
        offset += or_element_cnt;

        for (auto & [int_pid, pdat_ptr] : interfaces) {

            if (pdat_ptr->size() > 0) {

                sycl::buffer<T> buf_pcifield(pdat_ptr->data(),pdat_ptr->size());
                u32 len_pci = pdat_ptr->size();

                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = buf_pcifield.template get_access<sycl::access::mode::read>(cgh);
                    auto dest = buf->template get_access<sycl::access::mode::discard_write>(cgh);
                    auto off = offset;
                    cgh.parallel_for( sycl::range{len_pci}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                });
                offset += len_pci;

            }
        }

    }

};


template<class T>
inline void make_merge_patches_comp_field(
    SchedulerMPI & sched,
    PatchComputeField<T> & fields,
    PatchComputeFieldInterfaces<T> & interfaces,
    std::unordered_map<u64,IntMergedPatchComputeField<T>> & merge_pdat_buf){



    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
        std::cout << "patch : n°"<<id_patch << " -> merge compute field" << std::endl;
        merge_pdat_buf[id_patch].gen(fields.field_data[id_patch],interfaces.interface_map[id_patch]);
    });

}

 





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
        std::unordered_map<u64, f32> min_cfl_map;
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            u32 npart_patch = pdat_buf.get_pos<pos_vec>()->size();

            std::unique_ptr<sycl::buffer<f32>> buf_cfl = std::make_unique<sycl::buffer<f32>>(npart_patch);

            cl::sycl::range<1> range_npart{npart_patch};

            

            auto ker_Reduc_step_mincfl = [&](cl::sycl::handler &cgh) {

                auto arr = buf_cfl->get_access<sycl::access::mode::discard_write>(cgh);

                auto U1 = pdat_buf.get_U1<pos_prec>()->template get_access<sycl::access::mode::read>(cgh);
                auto U3 = pdat_buf.get_U3<pos_vec>()->template get_access<sycl::access::mode::read>(cgh);

                f32 cs = 1;

                constexpr u32 nvar_U3 = DU3::nvar;
                constexpr u32 iaxyz = DU3::iaxyz;

                constexpr u32 nvar_U1 = DU1::nvar;
                constexpr u32 ih = DU1::ihpart;

                constexpr f32 C_cour = 0.3;
                constexpr f32 C_force = 0.3;

                cgh.parallel_for<class Initial_dtcfl>( range_npart, [=](cl::sycl::item<1> item) {

                    u32 i = (u32) item.get_id(0);
                    
                    f32 h_a = U1[nvar_U1*i + ih];
                    f32_3 axyz = U3[nvar_U3*i + iaxyz];

                    f32 dtcfl_P = C_cour*h_a/cs;
                    f32 dtcfl_a = C_force*sycl::sqrt(h_a/sycl::length(axyz));

                    arr[i] = sycl::min(dtcfl_P,dtcfl_a);

                });

            };

            hndl.get_queue_compute(0).submit(ker_Reduc_step_mincfl);

            f32 min_cfl = syclalg::get_min<f32, 1, 0>(hndl.get_queue_compute(0), buf_cfl);

            min_cfl_map[id_patch] = min_cfl;
        });

        f32 cur_cfl_dt = HUGE_VALF;
        for (auto & [k,cfl] : min_cfl_map) {
            cur_cfl_dt = sycl::min(cur_cfl_dt,cfl);
        }

        std::cout << "node dt cfl : " << cur_cfl_dt << std::endl;

        f32 dt_cur;
        mpi::allreduce(&cur_cfl_dt, &dt_cur, 1, mpi_type_f32, MPI_MIN, MPI_COMM_WORLD);

        std::cout << " --- current dt  : " << dt_cur << std::endl;

        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

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

        /*
        std::cout << "chech no h null" << std::endl;
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "pid : " << id_patch << std::endl;

            std::cout << "cnt : " << pdat_buf.element_count << std::endl;

            auto U1 = 
                //merge_pdat_buf[id_patch].data.U1_s
                pdat_buf.U1_s
            ->template get_access<sycl::access::mode::read>();

            for (u32 i = 0; i < pdat_buf.element_count; i++) {

                f32 val = U1[i*2 + 0];
                if(val == 0){
                    std::cout << "----- fail id " << i  << " " << val << std::endl;
                    int a ;
                    std::cin >> a;
                }
            }
        });
        */

        
        std::cout << "particle reatribution" << std::endl;
        reatribute_particles(sched, sptree);

        /*

        std::cout << "chech no h null" << std::endl;
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "pid : " << id_patch << std::endl;

            std::cout << "cnt : " << pdat_buf.element_count << std::endl;
            auto U1 = 
                //merge_pdat_buf[id_patch].data.U1_s
                pdat_buf.U1_s
            ->template get_access<sycl::access::mode::read>();

            for (u32 i = 0; i < pdat_buf.element_count; i++) {

                f32 val = U1[i*2 + 0];
                if(val == 0){
                    std::cout << "----- fail id " << i  << " " << val << std::endl;
                    int a ;
                    std::cin >> a;
                }
            }
        });
        */







        constexpr pos_prec htol_up_tol = 1.2;


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
        std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> merge_pdat_buf;
        make_merge_patches(sched,interface_hndl, merge_pdat_buf);
        hndl.get_queue_compute(0).wait();
        tmerge_buf.stop();







        auto tgen_trees = timings::start_timer("radix tree compute", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, pos_vec>>> radix_trees;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {

            

            std::cout << "patch : n°"<<id_patch << " -> making radix tree" << std::endl;
            if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & mpdat_buf = merge_pdat_buf[id_patch].data;


            std::tuple<f32_3,f32_3> & box = merge_pdat_buf[id_patch].box; 

            //radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, pos_vec>>(hndl.get_queue_compute(0), box, mpdat_buf.get_pos<pos_vec>());
            
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;
            if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;
            radix_trees[id_patch]->compute_cellvolume(hndl.get_queue_compute(0));
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {

            std::cout << "patch : n°"<<id_patch << " -> radix tree compute interaction box" << std::endl;
            if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & mpdat_buf = merge_pdat_buf[id_patch].data;

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


        
        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°" << id_patch << "init h iter" << std::endl;
            if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & pdat_buf_merge = merge_pdat_buf[id_patch].data;
            
            sycl::buffer<f32> & hnew = *hnew_field.field_data_buf[id_patch];
            sycl::buffer<f32> & omega = *omega_field.field_data_buf[id_patch];
            sycl::buffer<f32> eps_h = sycl::buffer<f32>(merge_pdat_buf[id_patch].or_element_cnt);

            sycl::range range_npart{merge_pdat_buf[id_patch].or_element_cnt};

            std::cout << "   original size : " << merge_pdat_buf[id_patch].or_element_cnt << " | merged : " << pdat_buf_merge.element_count << std::endl;


            


            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                
                auto U1 = pdat_buf_merge.get_U1<f32>()->get_access<sycl::access::mode::read>(cgh);
                auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
                auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

                cgh.parallel_for<class Init_iterate_h>( range_npart, [=](sycl::item<1> item) {
                        
                    u32 id_a = (u32) item.get_id(0);

                    h[id_a] = U1[id_a*DU1::nvar + DU1::ihpart];
                    eps[id_a] = 100;

                });

            });



            
            for (u32 it_num = 0 ; it_num < 4; it_num++) {
                std::cout << "patch : n°" << id_patch << "h iter" << std::endl;
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                    auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                    auto eps = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                    auto U1 = pdat_buf_merge.get_U1<f32>()->get_access<sycl::access::mode::read>(cgh);
                    auto r = pdat_buf_merge.pos_s->get_access<sycl::access::mode::read>(cgh);
                    
                    using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                    Rta tree_acc(*radix_trees[id_patch], cgh);



                    auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                    f32 part_mass = gpart_mass;

                    constexpr f32 h_max_tot_max_evol = htol_up_tol;
                    constexpr f32 h_max_evol_p = htol_up_tol;
                    constexpr f32 h_max_evol_m = 1/htol_up_tol;

                    cgh.parallel_for<class SPHTest>(range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32)item.get_id(0);


                        if(eps[id_a] > 1e-4){

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
                            f32 omega_a = 1 + (h_a/(3*rho_ha))*part_omega_sum;
                            f32 new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

                            bool max_achieved = false;
                            if(new_h < h_a*h_max_evol_m) new_h = h_max_evol_m*h_a;
                            if(new_h > h_a*h_max_evol_p) new_h = h_max_evol_p*h_a;

                            
                            f32 ha_0 = U1[id_a*DU1::nvar + DU1::ihpart];
                            
                            
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




            std::cout << "patch : n°" << id_patch << "compute omega" << std::endl;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_buf_merge.pos_s->get_access<sycl::access::mode::read>(cgh);
                
                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(*radix_trees[id_patch], cgh);



                auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                f32 part_mass = gpart_mass;

                constexpr f32 h_max_tot_max_evol = htol_up_tol;
                constexpr f32 h_max_evol_p = htol_up_tol;
                constexpr f32 h_max_evol_m = 1/htol_up_tol;

                cgh.parallel_for<class write_omega>(range_npart, [=](sycl::item<1> item) {
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
            

            
            



            //write back h test
            //*
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);

                auto U1 = pdat_buf_merge.get_U1<f32>()->get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class write_back_h>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);

                    U1[id_a*DU1::nvar + DU1::ihpart] = h_new[id_a];

                });

            });
            //*/



        });

        //


        

        hnew_field.to_map();
        omega_field.to_map();

        std::cout << "echange interface hnew" << std::endl;
        PatchComputeFieldInterfaces<pos_prec> hnew_field_interfaces = interface_hndl.template comm_interfaces_field<pos_prec>(sched,hnew_field);
        std::cout << "echange interface omega" << std::endl;
        PatchComputeFieldInterfaces<pos_prec> omega_field_interfaces = interface_hndl.template comm_interfaces_field<pos_prec>(sched,omega_field);

        std::unordered_map<u64, IntMergedPatchComputeField<f32>> hnew_field_merged;
        make_merge_patches_comp_field<f32>(sched, hnew_field, hnew_field_interfaces, hnew_field_merged);
        std::unordered_map<u64, IntMergedPatchComputeField<f32>> omega_field_merged;
        make_merge_patches_comp_field<f32>(sched, omega_field, omega_field_interfaces, omega_field_merged);


        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            if(merge_pdat_buf[id_patch].or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & pdat_buf_merge = merge_pdat_buf[id_patch].data;
            
            sycl::buffer<f32> & hnew =  * hnew_field_merged[id_patch].buf;
            sycl::buffer<f32> & omega = * omega_field_merged[id_patch].buf;

            sycl::range range_npart{merge_pdat_buf[id_patch].or_element_cnt};

            std::cout << "patch : n°" << id_patch << "compute forces" << std::endl;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                auto omga = omega.get_access<sycl::access::mode::read>(cgh);

                auto r = pdat_buf_merge.pos_s->get_access<sycl::access::mode::read>(cgh);
                auto axyz = pdat_buf_merge.U3_s->get_access<sycl::access::mode::discard_write>(cgh);
                
                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(*radix_trees[id_patch], cgh);



                auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                f32 part_mass = gpart_mass;
                f32 cs = 1;

                constexpr f32 htol = htol_up_tol;


                cgh.parallel_for<class forces>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32) item.get_id(0);

                    f32_3 sum_axyz = {0,0,0};
                    f32 h_a = h_new[id_a];

                    f32_3 xyz_a = r[id_a];

                    f32 rho_a = rho_h(part_mass, h_a);
                    f32 rho_a_sq = rho_a*rho_a;

                    f32 P_a = cs*cs*rho_a;
                    f32 omega_a = omga[id_a];


                    f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                    f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                    
                    
                    walker::rtree_for(
                        tree_acc,
                        [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                            f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                            float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern * htol;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            //compute only omega_a
                            f32_3 dr = xyz_a - r[id_b];
                            f32 rab = sycl::length(dr);
                            f32 h_b = h_new[id_b];

                            if(rab > h_a*Kernel::Rkern && rab > h_b*Kernel::Rkern) return;

                            f32_3 r_ab_unit = dr / rab;

                            if(rab < 1e-9){
                                r_ab_unit = {0,0,0};
                            }

                            
                            f32 rho_b = rho_h(part_mass, h_b);
                            f32 P_b = cs*cs*rho_b;
                            f32 omega_b = omga[id_b];

                            sum_axyz += sph_pressure<f32_3,f32>(
                                part_mass, rho_a_sq, rho_b*rho_b, P_a, P_b, omega_a, omega_b
                                , 0,0, r_ab_unit*Kernel::dW(rab,h_a), r_ab_unit*Kernel::dW(rab,h_b));


                        },
                        [](u32 node_id) {});
                    

                    
                    axyz[id_a] = sum_axyz;
                    

                });

            }); 


            leapfrog_corrector<pos_prec, DU3>(
                hndl.get_queue_compute(0), 
                merge_pdat_buf[id_patch].or_element_cnt, 
                dt_cur, 
                pdat_buf_merge.get_pos<pos_vec>(), 
                pdat_buf_merge.get_U3<pos_vec>());

        });




        write_back_merge_patches(sched,interface_hndl, merge_pdat_buf);



        











        

        
    }




};



/*

hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

    auto h_new = hnew.get_access<sycl::access::mode::read_write>();

    auto U1 = pdat_buf.U1_s->get_access<sycl::access::mode::read>(cgh);
    auto r = pdat_buf.pos_s->get_access<sycl::access::mode::read>(cgh);
    walker::Radix_tree_accessor<u32, f32_3> tree_acc(*radix_trees[id_patch], cgh);



    auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

    

    cgh.parallel_for<class SPHTest>(sycl::range(pdat_buf.pos_s->size()), [=](sycl::item<1> item) {
        u32 id_a = (u32)item.get_id(0);

        f32_3 xyz_a = r[id_a]; // could be recovered from lambda

        f32 h_a = h_new[id_a*DU1::nvar + DU1::ihpart];

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