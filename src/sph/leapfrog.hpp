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
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/patchdata_field.hpp"
#include "patch/serialpatchtree.hpp"
#include "patch/merged_patch.hpp"
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
#include "forces.hpp"


constexpr f32 gpart_mass =2e-4;


template<class pos_vec>
inline void dump_merged_patches(std::string prefix, SchedulerMPI & sched,std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merged_map){
    struct PatchFile{
        MPI_File mfile;
        std::string name;
    };

    std::unordered_map<u64, u64> pfile_map;
    std::vector<PatchFile> patch_files(sched.patch_list.global.size());

    for(u32 i = 0 ; i < sched.patch_list.global.size(); i++){
        pfile_map[sched.patch_list.global[i].id_patch] = i;
        patch_files[i].name = prefix + "patchdata_merged_" + std::to_string(sched.patch_list.global[i].id_patch) + ".bin";
    }
    
    for(PatchFile & pf : patch_files){

        //std::cout << "opening : " << pf.name << std::endl;
        int rc = mpi::file_open(MPI_COMM_WORLD, pf.name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &pf.mfile);

        if (rc) {
            printf( "Unable to open file \"%s\"\n", pf.name.c_str() );fflush(stdout);
        }
    }
    

    {

        for(auto & [pid,pdat] : merged_map){
            

            MPI_File & mfilepatch = patch_files[pfile_map[pid]].mfile;


            PatchData tmp(pdat.data->pdl);
            {
                PatchDataBuffer & pdat_buf = * pdat.data;



                for(u32 idx = 0; idx < pdat_buf.fields_f32.size(); idx++){
                    tmp.fields_f32[idx].resize( pdat_buf.fields_f32[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_2.size(); idx++){
                    tmp.fields_f32_2[idx].resize( pdat_buf.fields_f32_2[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_3.size(); idx++){
                    tmp.fields_f32_3[idx].resize( pdat_buf.fields_f32_3[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_4.size(); idx++){
                    tmp.fields_f32_4[idx].resize( pdat_buf.fields_f32_4[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_8.size(); idx++){
                    tmp.fields_f32_8[idx].resize( pdat_buf.fields_f32_8[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_16.size(); idx++){
                    tmp.fields_f32_16[idx].resize( pdat_buf.fields_f32_16[idx]->size() );
                }



                for(u32 idx = 0; idx < pdat_buf.fields_f64.size(); idx++){
                    tmp.fields_f64[idx].resize( pdat_buf.fields_f64[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_2.size(); idx++){
                    tmp.fields_f64_2[idx].resize( pdat_buf.fields_f64_2[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_3.size(); idx++){
                    tmp.fields_f64_3[idx].resize( pdat_buf.fields_f64_3[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_4.size(); idx++){
                    tmp.fields_f64_4[idx].resize( pdat_buf.fields_f64_4[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_8.size(); idx++){
                    tmp.fields_f64_8[idx].resize( pdat_buf.fields_f64_8[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_16.size(); idx++){
                    tmp.fields_f64_16[idx].resize( pdat_buf.fields_f64_16[idx]->size() );
                }


                for(u32 idx = 0; idx < pdat_buf.fields_u32.size(); idx++){
                    tmp.fields_u32[idx].resize( pdat_buf.fields_u32[idx]->size() );
                }

                for(u32 idx = 0; idx < pdat_buf.fields_u64.size(); idx++){
                    tmp.fields_u64[idx].resize( pdat_buf.fields_u64[idx]->size() );
                }



                for(u32 idx = 0; idx < pdat_buf.fields_f32.size(); idx++){
                    auto acc = pdat_buf.fields_f32[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32[idx]->size(); i++) { tmp.fields_f32[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_2.size(); idx++){
                    auto acc = pdat_buf.fields_f32_2[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_2[idx]->size(); i++) { tmp.fields_f32_2[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_3.size(); idx++){
                    auto acc = pdat_buf.fields_f32_3[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_3[idx]->size(); i++) { tmp.fields_f32_3[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_4.size(); idx++){
                    auto acc = pdat_buf.fields_f32_4[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_4[idx]->size(); i++) { tmp.fields_f32_4[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_8.size(); idx++){
                    auto acc = pdat_buf.fields_f32_8[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_8[idx]->size(); i++) { tmp.fields_f32_8[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_16.size(); idx++){
                    auto acc = pdat_buf.fields_f32_16[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_16[idx]->size(); i++) { tmp.fields_f32_16[idx].data()[i] = acc[i];}
                }




                for(u32 idx = 0; idx < pdat_buf.fields_f64.size(); idx++){
                    auto acc = pdat_buf.fields_f64[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64[idx]->size(); i++) { tmp.fields_f64[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_2.size(); idx++){
                    auto acc = pdat_buf.fields_f64_2[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_2[idx]->size(); i++) { tmp.fields_f64_2[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_3.size(); idx++){
                    auto acc = pdat_buf.fields_f64_3[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_3[idx]->size(); i++) { tmp.fields_f64_3[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_4.size(); idx++){
                    auto acc = pdat_buf.fields_f64_4[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_4[idx]->size(); i++) { tmp.fields_f64_4[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_8.size(); idx++){
                    auto acc = pdat_buf.fields_f64_8[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_8[idx]->size(); i++) { tmp.fields_f64_8[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_16.size(); idx++){
                    auto acc = pdat_buf.fields_f64_16[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_16[idx]->size(); i++) { tmp.fields_f64_16[idx].data()[i] = acc[i];}
                }




                for(u32 idx = 0; idx < pdat_buf.fields_u32.size(); idx++){
                    auto acc = pdat_buf.fields_u32[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_u32[idx]->size(); i++) { tmp.fields_u32[idx].data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_u64.size(); idx++){
                    auto acc = pdat_buf.fields_u64[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_u64[idx]->size(); i++) { tmp.fields_u64[idx].data()[i] = acc[i];}
                }


            }



            



            file_write_patchdata(mfilepatch, tmp);

        }


        
    }

    for(PatchFile & pf : patch_files){
        mpi::file_close(&pf.mfile);
    }
}














/*

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
*/





template <class flt>
inline void leapfrog_predictor(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_vxyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_axyz) {

    using vec3 = sycl::vec<flt, 3>;

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto acc_xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_vxyz  = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_axyz  = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 & vxyz = acc_vxyz[item];
            vec3 & axyz = acc_axyz[item];

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

template <class flt>
inline void leapfrog_corrector(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_vxyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_axyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_axyz_old){

    sycl::range<1> range_npart{npart};

    using vec3 = sycl::vec<flt, 3>;

    auto ker_corect_step = [&](sycl::handler &cgh) {
            
        auto acc_vxyz  = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_axyz  = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_axyz_old  = buf_axyz_old->template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(
            range_npart, [=](sycl::item<1> item) {
                
                u32 gid = (u32) item.get_id();

                vec3 & vxyz = acc_vxyz[item];
                vec3 & axyz = acc_axyz[item];
                vec3 & axyz_old = acc_axyz_old[item];
    
                //v^* = v^{n + 1/2} + dt/2 a^n
                vxyz = vxyz + (dt/2) * (axyz - axyz_old);

            }
        );

    };

    queue.submit(ker_corect_step);

}


template<class flt>
inline void swap_a_field(sycl::queue &queue, u32 npart,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_axyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_axyz_old){
    sycl::range<1> range_npart{npart};

    using vec3 = sycl::vec<flt, 3>;

    auto ker_swap_a = [&](sycl::handler &cgh) {
            

        auto acc_axyz  = buf_axyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_axyz_old  = buf_axyz_old->template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(
            range_npart, [=](sycl::item<1> item) {
                

                vec3 axyz = acc_axyz[item];
                vec3 axyz_old = acc_axyz_old[item];
    
                acc_axyz[item] = axyz_old;
                acc_axyz_old[item] = axyz;

            }
        );

    };

    queue.submit(ker_swap_a);
}


template<class flt>
inline void position_modulo(sycl::queue &queue, u32 npart, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box){

    using vec3 = sycl::vec<flt, 3>;

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);
        vec3 delt = box_max - box_min;

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 r = xyz[gid] - box_min;

            r = sycl::fmod(r,delt);
            r+= delt;
            r = sycl::fmod(r,delt);
            r+= box_min;

            xyz[gid] = r;

        });
    };

    queue.submit(ker_predict_step);

}


/*
inline void print_interf_status(InterfaceHandler<f32_3, f32> & interf_hndl,PatchComputeFieldInterfaces<f32> & pcfi){
    //for(auto & [id_pr, dat_vec] : interf_hndl.interface_map){
        for(auto & [id_ps, pdat] : (interf_hndl.interface_map[0])){
            std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->U1_s.size() << std::endl;
        }
    //}

    std::cout << std::endl;

    //for(auto & [id_pr, dat_vec] : pcfi.interface_map){
        for(auto & [id_ps, pdat] : (pcfi.interface_map[0])){
            std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
        }
    //}
}
*/



/*
inline void dump_interf(std::string dump_pref, InterfaceHandler<f32_3, f32> & interf_hndl,PatchComputeFieldInterfaces<f32> & pcfi){

    //print_interf_status(interf_hndl,pcfi);


    u32 ii = 0;
    for(auto & [id_pr, dat_vec] : interf_hndl.interface_map){
        for(u32 idx = 0; idx < dat_vec.size() ; idx ++){

            u32 id_ps = std::get<0>(dat_vec[idx]);

            std::string fdump = dump_pref + std::to_string(id_pr) + "-" + std::to_string(id_ps) +"__"+std::to_string(ii)+ ".bin";

            std::cout << "dump interf : " << fdump << std::endl;

            std::unique_ptr<std::vector<f32>> & hf = std::get<1>(pcfi.interface_map[id_pr][idx]);
            std::unique_ptr<PatchData> & pdat = std::get<1>(dat_vec[idx]);

            std::cout << "-> " <<  pdat->U1_s.size() << " : " <<hf->size() << std::endl;

            PatchData pdat_w;

            pdat_w.pos_s.resize(pdat->pos_s.size());
            pdat_w.pos_d.resize(pdat->pos_d.size());
            pdat_w.U1_s.resize(pdat->U1_s.size());
            pdat_w.U1_d.resize(pdat->U1_d.size());
            pdat_w.U3_s.resize(pdat->U3_s.size());
            pdat_w.U3_d.resize(pdat->U3_d.size());

            for (u32 i = 0; i < hf->size(); i++) {
                pdat_w.U1_s[i*2 + 0] = (*hf)[i];
            }

            u32 sz_pref[6];

            sz_pref[0] = pdat_w.pos_s.size();
            sz_pref[1] = pdat_w.pos_d.size();
            sz_pref[2] = pdat_w.U1_s.size() ;
            sz_pref[3] = pdat_w.U1_d.size() ;
            sz_pref[4] = pdat_w.U3_s.size() ;
            sz_pref[5] = pdat_w.U3_d.size() ;

            MPI_File mfile;
            mpi::file_open(MPI_COMM_WORLD,fdump.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &mfile);

            MPI_Status st;
            mpi::file_write(mfile,sz_pref, 6, mpi_type_u32, &st);

            mpi::file_write(mfile, pdat_w.pos_s.data(), sz_pref[0] , mpi_type_f32_3, &st);
            mpi::file_write(mfile, pdat_w.pos_d.data(), sz_pref[1] , mpi_type_f64_3, &st);
            mpi::file_write(mfile, pdat_w.U1_s.data() , sz_pref[2] , mpi_type_f32  , &st);
            mpi::file_write(mfile, pdat_w.U1_d.data() , sz_pref[3] , mpi_type_f64  , &st);
            mpi::file_write(mfile, pdat_w.U3_s.data() , sz_pref[4] , mpi_type_f32_3, &st);
            mpi::file_write(mfile, pdat_w.U3_d.data() , sz_pref[5] , mpi_type_f64_3, &st);

            mpi::file_close(&mfile);

            ii++;

        }
    }

}
*/









template<class pos_prec>
class SPHTimestepperLeapfrog{public:

    
    using pos_vec  = sycl::vec<pos_prec, 3>;

    using u_morton = u32;

    using Kernel = sph::kernels::M4<f32>;


    inline void step(SchedulerMPI &sched,std::string dump_folder,u32 step_cnt,f64 & step_time){


        bool periodic_bc = true;


        SyCLHandler &hndl = SyCLHandler::get_instance();


        SerialPatchTree<pos_vec> sptree(sched.patch_tree, sched.get_box_tranform<pos_vec>());
        sptree.attach_buf();


        const u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
        const u32 ivxyz = sched.pdl.get_field_idx<f32_3>("vxyz");
        const u32 iaxyz = sched.pdl.get_field_idx<f32_3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<f32_3>("axyz_old");

        const u32 ihpart = sched.pdl.get_field_idx<f32>("hpart");


        //cfl
        std::unordered_map<u64, f32> min_cfl_map;
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            u32 npart_patch = pdat_buf.element_count;

            std::cout << "npart_patch : " << npart_patch << std::endl;

            std::unique_ptr<sycl::buffer<f32>> buf_cfl = std::make_unique<sycl::buffer<f32>>(npart_patch);

            sycl::range<1> range_npart{npart_patch};

            

            auto ker_Reduc_step_mincfl = [&](sycl::handler &cgh) {

                auto arr = buf_cfl->get_access<sycl::access::mode::discard_write>(cgh);

                auto acc_hpart = pdat_buf.fields_f32[ihpart]->template get_access<sycl::access::mode::read>(cgh);
                auto acc_axyz = pdat_buf.fields_f32_3[iaxyz]->template get_access<sycl::access::mode::read>(cgh);

                f32 cs = 1;

                constexpr f32 C_cour = 0.1;
                constexpr f32 C_force = 0.1;

                cgh.parallel_for<class Initial_dtcfl>( range_npart, [=](sycl::item<1> item) {

                    u32 i = (u32) item.get_id(0);
                    
                    f32 h_a = acc_hpart[item];
                    f32_3 axyz = acc_axyz[item];

                    f32 dtcfl_P = C_cour*h_a/cs;
                    f32 dtcfl_a = C_force*sycl::sqrt(h_a/sycl::length(axyz));

                    arr[i] = sycl::min(dtcfl_P,dtcfl_a);

                });

            };

            hndl.get_queue_compute(0).submit(ker_Reduc_step_mincfl);

            f32 min_cfl = syclalg::get_min<f32>(hndl.get_queue_compute(0), buf_cfl);

            min_cfl_map[id_patch] = min_cfl;
        });

        f32 cur_cfl_dt = 0.001;
        for (auto & [k,cfl] : min_cfl_map) {
            cur_cfl_dt = sycl::min(cur_cfl_dt,cfl);
        }

        std::cout << "node dt cfl : " << cur_cfl_dt << std::endl;

        f32 dt_cur;
        mpi::allreduce(&cur_cfl_dt, &dt_cur, 1, mpi_type_f32, MPI_MIN, MPI_COMM_WORLD);

        std::cout << " --- current dt  : " << dt_cur << std::endl;

        step_time += dt_cur;

        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

            std::cout << "patch : n°"<<id_patch << " -> leapfrog predictor" << std::endl;

            leapfrog_predictor<pos_prec>(
                hndl.get_queue_compute(0), 
                pdat_buf.element_count, 
                dt_cur, 
                pdat_buf.fields_f32_3.at(ixyz), 
                pdat_buf.fields_f32_3.at(ivxyz), 
                pdat_buf.fields_f32_3.at(iaxyz));

            std::cout << "patch : n°"<<id_patch << " -> a field swap" << std::endl;

            swap_a_field<pos_prec>(
                hndl.get_queue_compute(0), 
                pdat_buf.element_count, 
                pdat_buf.fields_f32_3.at(iaxyz), 
                pdat_buf.fields_f32_3.at(iaxyz_old));

            if (periodic_bc) {
                position_modulo<pos_prec>(hndl.get_queue_compute(0), pdat_buf.element_count,  
                pdat_buf.fields_f32_3[ixyz], sched.get_box_volume<pos_vec>());
            }

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

        //std::filesystem::create_directory("step_before_reatrib" + std::to_string(step_cnt));
        //std::cout << "dumping state"<<std::endl;
        //dump_state("step_before_reatrib" + std::to_string(step_cnt) + "/", sched,step_time);
        
        std::cout << "particle reatribution" << std::endl;
        reatribute_particles(sched, sptree,periodic_bc);

        //std::filesystem::create_directory("step_after_reatrib" + std::to_string(step_cnt));
        //std::cout << "dumping state"<<std::endl;
        //dump_state("step_after_reatrib" + std::to_string(step_cnt) + "/", sched,step_time);

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







        constexpr pos_prec htol_up_tol = 1.4;
        constexpr pos_prec htol_up_iter = 1.2;


        std::cout << "exhanging interfaces" << std::endl;
        auto timer_h_max = timings::start_timer("compute_hmax", timings::timingtype::function);

        std::cout << "owned_data : " << std::endl;
        for (auto & [k,a] : sched.patch_data.owned_data) {
            std::cout << " pdat : " << k << std::endl;
        }

        PatchField<f32> h_field;
        sched.compute_patch_field(
            h_field, 
            get_mpi_type<pos_prec>(), 
            [htol_up_tol](sycl::queue & queue, Patch & p, PatchDataBuffer & pdat_buf){
                return patchdata::sph::get_h_max<pos_prec>(pdat_buf.pdl,queue, pdat_buf)*htol_up_tol*Kernel::Rkern;
            }
        );

        timer_h_max.stop();




        InterfaceHandler<pos_vec, pos_prec> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<pos_vec, pos_prec>>(sched, sptree, h_field,periodic_bc);
        interface_hndl.comm_interfaces(sched,periodic_bc);





        //merging strat
        auto tmerge_buf = timings::start_timer("buffer merging", timings::sycl);
        std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> merge_pdat_buf;
        make_merge_patches(sched,interface_hndl, merge_pdat_buf);
        hndl.get_queue_compute(0).wait();
        tmerge_buf.stop();

        //dump_merged_patches(dump_folder+"/merged0_", sched, merge_pdat_buf);





        auto tgen_trees = timings::start_timer("radix tree compute", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, pos_vec>>> radix_trees;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {

            

            std::cout << "patch : n°"<<id_patch << " -> making radix tree" << std::endl;
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & mpdat_buf = * merge_pdat_buf.at(id_patch).data;


            std::tuple<f32_3,f32_3> & box =merge_pdat_buf.at(id_patch).box; 

            //radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, pos_vec>>(hndl.get_queue_compute(0), box, mpdat_buf.fields_f32_3[ixyz]);
            
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            std::cout << "patch : n°"<<id_patch << " -> radix tree compute volume" << std::endl;
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;
            radix_trees[id_patch]->compute_cellvolume(hndl.get_queue_compute(0));
        });


        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {

            std::cout << "patch : n°"<<id_patch << " -> radix tree compute interaction box" << std::endl;
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & mpdat_buf = * merge_pdat_buf.at(id_patch).data;

            radix_trees[id_patch]->compute_int_boxes(hndl.get_queue_compute(0),mpdat_buf.fields_f32[ihpart],htol_up_tol);

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
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & pdat_buf_merge = * merge_pdat_buf.at(id_patch).data;
            
            sycl::buffer<f32> & hnew  = *hnew_field.field_data_buf[id_patch];
            sycl::buffer<f32> & omega = *omega_field.field_data_buf[id_patch];
            sycl::buffer<f32> eps_h   = sycl::buffer<f32>(merge_pdat_buf.at(id_patch).or_element_cnt);

            sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

            std::cout << "   original size : " <<merge_pdat_buf.at(id_patch).or_element_cnt << " | merged : " << pdat_buf_merge.element_count << std::endl;


            sph::algs::SmoothingLenghtCompute<f32, u32, Kernel> h_iterator(sched.pdl,htol_up_tol,htol_up_iter);


            sycl_init_h_iter_bufs(
                hndl.get_queue_compute(0), 
                merge_pdat_buf.at(id_patch).or_element_cnt, 
                ihpart, 
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);



        
            for (u32 it_num = 0 ; it_num < 30; it_num++) {
                //std::cout << "patch : n°" << id_patch << "h iter" << std::endl;
                
                sycl_h_iter_step(hndl.get_queue_compute(0), 
                    merge_pdat_buf.at(id_patch).or_element_cnt, 
                    ihpart, 
                    ixyz,
                    gpart_mass,
                    htol_up_tol,
                    htol_up_iter,
                    *radix_trees[id_patch],
                    pdat_buf_merge, 
                    hnew, 
                    omega, 
                    eps_h);

            }




            std::cout << "patch : n°" << id_patch << "compute omega" << std::endl;
            sycl_h_iter_omega(hndl.get_queue_compute(0), 
                merge_pdat_buf.at(id_patch).or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                *radix_trees[id_patch],
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);
            
            

            
            



            //write back h test
            //*
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);

                auto acc_hpart = pdat_buf_merge.fields_f32.at(ihpart)->get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class write_back_h>(range_npart, [=](sycl::item<1> item) {
                    acc_hpart[item] = h_new[item];
                });

            });
            //*/



        });

        //

        //dump_merged_patches(dump_folder+"/merged05_", sched, merge_pdat_buf);
        

        hnew_field.to_map();
        omega_field.to_map();

        std::cout << "echange interface hnew" << std::endl;
        PatchComputeFieldInterfaces<pos_prec> hnew_field_interfaces = interface_hndl.template comm_interfaces_field<pos_prec>(sched,hnew_field,periodic_bc);
        std::cout << "echange interface omega" << std::endl;
        PatchComputeFieldInterfaces<pos_prec> omega_field_interfaces = interface_hndl.template comm_interfaces_field<pos_prec>(sched,omega_field,periodic_bc);


        //dump_interf(dump_folder+"/interf_",interface_hndl,hnew_field_interfaces);


        hnew_field.to_sycl();
        omega_field.to_sycl();

        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> hnew_field_merged;
        make_merge_patches_comp_field<f32>(sched,  interface_hndl,hnew_field, hnew_field_interfaces, hnew_field_merged);
        std::unordered_map<u64, MergedPatchCompFieldBuffer<f32>> omega_field_merged;
        make_merge_patches_comp_field<f32>(sched,  interface_hndl, omega_field, omega_field_interfaces, omega_field_merged);

        //*

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & pdat_buf_merge = * merge_pdat_buf.at(id_patch).data;
            
            sycl::buffer<f32> & hnew =  * hnew_field_merged[id_patch].buf;
            sycl::buffer<f32> & omega = * omega_field_merged[id_patch].buf;

            std::cout << "patch : n°" << id_patch << "write back merged" << std::endl;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                sycl::range range_npart{hnew.size()};

                auto hw = pdat_buf_merge.fields_f32[ihpart]->get_access<sycl::access::mode::write>(cgh);
                auto hr = hnew.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(range_npart, [=](sycl::item<1> item) {

                    hw[item] = hr[item];

                });



            });

        });


        //dump_merged_patches(dump_folder+"/merged1_", sched, merge_pdat_buf);
        //*/

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;

            PatchDataBuffer & pdat_buf_merge = * merge_pdat_buf.at(id_patch).data;
            
            sycl::buffer<f32> & hnew =  * hnew_field_merged[id_patch].buf;
            sycl::buffer<f32> & omega = * omega_field_merged[id_patch].buf;

            sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

            
            if(step_cnt > 4){
                std::cout << "patch : n°" << id_patch << "compute forces" << std::endl;
                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {

                    auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                    auto omga = omega.get_access<sycl::access::mode::read>(cgh);

                    auto r = pdat_buf_merge.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);
                    auto acc_axyz = pdat_buf_merge.fields_f32_3[iaxyz]->get_access<sycl::access::mode::discard_write>(cgh);
                    
                    using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                    Rta tree_acc(*radix_trees[id_patch], cgh);



                    auto cell_int_r = radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                    f32 part_mass = gpart_mass;
                    f32 cs = 1;

                    constexpr f32 htol = htol_up_tol;


                    //sycl::stream out(65000,65000,cgh);

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


                                f32_3 tmp = sph_pressure<f32_3,f32>(
                                    part_mass, rho_a_sq, rho_b*rho_b, P_a, P_b, omega_a, omega_b
                                    , 0,0, r_ab_unit*Kernel::dW(rab,h_a), r_ab_unit*Kernel::dW(rab,h_b));


                                //out << omega_a<< " " << tmp << "\n";

                                sum_axyz += tmp;

                            },
                            [](u32 node_id) {});
                        
                        //out << "sum : " << sum_axyz << "\n";
                        
                        acc_axyz[id_a] = sum_axyz;
                        

                    });

                }); 
            }
/*
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto U3 = pdat_buf_merge.U3_s->get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class forces>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32) item.get_id(0);

                    U3[id_a*DU3::nvar + DU3::iaxyz] -= 2.f*U3[id_a*DU3::nvar + DU3::ivxyz] ;

                });
            });

            
*/

            if(step_cnt > 5){

                std::cout << "leapfrog corrector " << std::endl;

                leapfrog_corrector<pos_prec>(
                    hndl.get_queue_compute(0), 
                   merge_pdat_buf.at(id_patch).or_element_cnt, 
                    dt_cur, 
                    pdat_buf_merge.fields_f32_3[ivxyz], 
                    pdat_buf_merge.fields_f32_3[iaxyz],
                    pdat_buf_merge.fields_f32_3[iaxyz_old]);
            }
            /*
            std::cout << "omega access corrector " << std::endl;
            auto tmp_acc = omega.get_access<sycl::access::mode::read>();

            
            std::cout << "exemple val " << tmp_acc[0] << " "
              << tmp_acc[0] << " "
              << tmp_acc[50] << " "
              << tmp_acc[1000] << " "
              << tmp_acc[3452] << " "
              << tmp_acc[8888] << " "
              << tmp_acc[25000] << " "
              << tmp_acc[8719] << " "
              << tmp_acc[819] << " "
              << tmp_acc[4541] << std::endl;
              //*/
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