#pragma once
#include "aliases.hpp"
#include "io/logs.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_layout.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include <stdexcept>
#include <sys/mpi_handler.hpp>
#include <string>
#include <unordered_map>
#include <vector>

inline void dump_patch_data(std::string prefix, SchedulerMPI & sched){

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

        //std::cout << "opening : " << pf.name << std::endl;
        int rc = mpi::file_open(MPI_COMM_WORLD, pf.name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &pf.mfile);

        if (rc) {
            printf( "Unable to open file \"%s\"\n", pf.name.c_str() );fflush(stdout);
        }
    }
    

    {

        for(auto & [pid,pdat] : sched.patch_data.owned_data){
            
            MPI_Status st;
            MPI_File & mfilepatch = patch_files[pfile_map[pid]].mfile;

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








            for(auto a : pdat.fields_f32){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f32_2){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f32_3){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f32_4){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f32_8){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f32_16){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }









            for(auto a : pdat.fields_f64){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f64_2){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f64_3){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f64_4){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f64_8){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_f64_16){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }





            for(auto a : pdat.fields_u32){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }

            for(auto a : pdat.fields_u64){
                u32 sz = a.size();
                using T = decltype(a)::Field_type;
                mpi::file_write(mfilepatch,a.data(), 1, get_mpi_type<T>(), &st);
            }


        }


        
    }

    for(PatchFile & pf : patch_files){
        mpi::file_close(&pf.mfile);
    }
}

inline void dump_patch_list(std::string prefix, SchedulerMPI & sched){
    

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

inline void dump_simbox(std::string prefix, SchedulerMPI & sched){
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

inline void dump_state(std::string prefix, SchedulerMPI & sched, f64 time){

    auto t = timings::start_timer("dump_state", timings::timingtype::function);

    dump_patch_data(prefix, sched);

    dump_patch_list(prefix, sched);
    dump_simbox(prefix, sched);
    dump_siminfo(prefix,time);

    t.stop();

}