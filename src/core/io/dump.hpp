// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
#include "aliases.hpp"
#include "core/io/logs.hpp"
#include "core/patch/merged_patch.hpp"
#include "core/patch/base/patch.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include <stdexcept>
#include "core/sys/mpi_handler.hpp"
#include <string>
#include <unordered_map>
#include <vector>
































inline void file_write_patchdata(MPI_File & mfilepatch , PatchData & pdat){

    MPI_Status st;

    std::string head = "##header start##\n";
    head.resize(16);
    mpi::file_write(mfilepatch,head.c_str(), 16, mpi_type_u8, &st);

    head = "#f32";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f32_2";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32_2){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f32_3";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32_3){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f32_4";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32_4){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f32_8";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32_8){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f32_16";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f32_16){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }







    head = "#f64";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f64_2";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64_2){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f64_3";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64_3){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f64_4";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64_4){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f64_8";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64_8){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#f64_16";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_f64_16){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }




    head = "#u32";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_u32){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }

    head = "#u64";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    for(auto a : pdat.fields_u64){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shamrock_exc("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    }


    head = "##header end##\n";
    head.resize(16);
    mpi::file_write(mfilepatch,head.c_str(), 16, mpi_type_u8, &st);








    for(auto a : pdat.fields_f32){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f32_2){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f32_3){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f32_4){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f32_8){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f32_16){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }









    for(auto a : pdat.fields_f64){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f64_2){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f64_3){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f64_4){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f64_8){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_f64_16){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }





    for(auto a : pdat.fields_u32){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

    for(auto a : pdat.fields_u64){
        u32 sz = a.size();
        using T = decltype(a)::Field_type;
        mpi::file_write(mfilepatch,a.usm_data(), sz, get_mpi_type<T>(), &st);
    }

}

inline void dump_patch_data(std::string prefix, PatchScheduler & sched){

    struct PatchFile{
        MPI_File mfile;
        std::string name;
    };

    std::unordered_map<u64, u64> pfile_map;
    std::vector<PatchFile> patch_files(sched.patch_list.global.size());

    for(u32 i = 0 ; i < sched.patch_list.global.size(); i++){
        pfile_map[sched.patch_list.global[i].id_patch] = i;
        patch_files[i].name = prefix + "patchdata_" + std::to_string(sched.patch_list.global[i].id_patch) + ".bin";
    }
    
    for(PatchFile & pf : patch_files){

        std::cout << "opening : " << pf.name << std::endl;
        int rc = mpi::file_open(MPI_COMM_WORLD, pf.name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &pf.mfile);

        if (rc) {
            printf( "Unable to open file \"%s\"\n", pf.name.c_str() );fflush(stdout);
        }
    }
    

    {

        for(auto & [pid,pdat] : sched.patch_data.owned_data){

            std::cout << "[" << mpi_handler::world_rank << "] writing pdat : " << pid << std::endl;
            
            MPI_Status st;
            MPI_File & mfilepatch = patch_files[pfile_map[pid]].mfile;


            file_write_patchdata(mfilepatch, pdat);

        }


        
    }

    for(PatchFile & pf : patch_files){
        mpi::file_close(&pf.mfile);
    }
}

inline void dump_patch_list(std::string prefix, PatchScheduler & sched){
    

    MPI_File patch_list_file;
    std::string fname = prefix + "patch_list.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &patch_list_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(mpi_handler::world_rank == 0){

        if(sched.patch_list.global.size() > u64(u32_max)){
            throw shamrock_exc("patch list size > u32_max not handled by dump");
        }

        MPI_Status st;
        mpi::file_write(patch_list_file, sched.patch_list.global.data(), sched.patch_list.global.size(), patch::patch_MPI_type,&st);
    }



    mpi::file_close(&patch_list_file);
}

inline void dump_simbox(std::string prefix, PatchScheduler & sched){
    MPI_File simbox_file;
    std::string fname = prefix + "simbox.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &simbox_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(mpi_handler::world_rank == 0){


        MPI_Status st;

        if(sched.pdl.xyz_mode == xyz32){
            u8 f = 0;
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.min_box_sim_s, 1, mpi_type_f32_3,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.max_box_sim_s, 1, mpi_type_f32_3,&st);
        }else if (sched.pdl.xyz_mode == xyz64){
            u8 f = 1;
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.min_box_sim_d, 1, mpi_type_f64_3,&st);    
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.max_box_sim_d, 1, mpi_type_f64_3,&st);       
        }
        
    }



    mpi::file_close(&simbox_file);
}

inline void dump_siminfo(std::string prefix, f64 time){
    MPI_File timeval_file;
    std::string fname = prefix + "timeval.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &timeval_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(mpi_handler::world_rank == 0){


        MPI_Status st;
        mpi::file_write(timeval_file, &time, 1,mpi_type_f64,&st);

        
    }



    mpi::file_close(&timeval_file);
}

inline void dump_state(std::string prefix, PatchScheduler & sched, f64 time){

    auto t = timings::start_timer("dump_state", timings::timingtype::function);

    dump_patch_data(prefix, sched);

    dump_patch_list(prefix, sched);
    dump_simbox(prefix, sched);
    dump_siminfo(prefix,time);

    t.stop();

}








// dirty /////////////////



template<class pos_vec>
inline void dump_merged_patches(std::string prefix, PatchScheduler & sched,std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merged_map){
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
                    for (u32 i = 0; i < pdat_buf.fields_f32[idx]->size(); i++) { tmp.fields_f32[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_2.size(); idx++){
                    auto acc = pdat_buf.fields_f32_2[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_2[idx]->size(); i++) { tmp.fields_f32_2[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_3.size(); idx++){
                    auto acc = pdat_buf.fields_f32_3[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_3[idx]->size(); i++) { tmp.fields_f32_3[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_4.size(); idx++){
                    auto acc = pdat_buf.fields_f32_4[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_4[idx]->size(); i++) { tmp.fields_f32_4[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_8.size(); idx++){
                    auto acc = pdat_buf.fields_f32_8[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_8[idx]->size(); i++) { tmp.fields_f32_8[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f32_16.size(); idx++){
                    auto acc = pdat_buf.fields_f32_16[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f32_16[idx]->size(); i++) { tmp.fields_f32_16[idx].usm_data()[i] = acc[i];}
                }




                for(u32 idx = 0; idx < pdat_buf.fields_f64.size(); idx++){
                    auto acc = pdat_buf.fields_f64[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64[idx]->size(); i++) { tmp.fields_f64[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_2.size(); idx++){
                    auto acc = pdat_buf.fields_f64_2[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_2[idx]->size(); i++) { tmp.fields_f64_2[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_3.size(); idx++){
                    auto acc = pdat_buf.fields_f64_3[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_3[idx]->size(); i++) { tmp.fields_f64_3[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_4.size(); idx++){
                    auto acc = pdat_buf.fields_f64_4[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_4[idx]->size(); i++) { tmp.fields_f64_4[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_8.size(); idx++){
                    auto acc = pdat_buf.fields_f64_8[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_8[idx]->size(); i++) { tmp.fields_f64_8[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_f64_16.size(); idx++){
                    auto acc = pdat_buf.fields_f64_16[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_f64_16[idx]->size(); i++) { tmp.fields_f64_16[idx].usm_data()[i] = acc[i];}
                }




                for(u32 idx = 0; idx < pdat_buf.fields_u32.size(); idx++){
                    auto acc = pdat_buf.fields_u32[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_u32[idx]->size(); i++) { tmp.fields_u32[idx].usm_data()[i] = acc[i];}
                }

                for(u32 idx = 0; idx < pdat_buf.fields_u64.size(); idx++){
                    auto acc = pdat_buf.fields_u64[idx]->get_access<sycl::access::mode::read>();
                    for (u32 i = 0; i < pdat_buf.fields_u64[idx]->size(); i++) { tmp.fields_u64[idx].usm_data()[i] = acc[i];}
                }


            }



            



            file_write_patchdata(mfilepatch, tmp);

        }


        
    }

    for(PatchFile & pf : patch_files){
        mpi::file_close(&pf.mfile);
    }
}



//////////////////////////