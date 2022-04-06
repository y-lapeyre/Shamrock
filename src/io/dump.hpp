#pragma once
#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
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
            

            MPI_File & mfilepatch = patch_files[pfile_map[pid]].mfile;

            u32 sz_pref[6];

            sz_pref[0] = pdat.pos_s.size();
            sz_pref[1] = pdat.pos_d.size();
            sz_pref[2] = pdat.U1_s.size() ;
            sz_pref[3] = pdat.U1_d.size() ;
            sz_pref[4] = pdat.U3_s.size() ;
            sz_pref[5] = pdat.U3_d.size() ;

            // std::cout << "writing "<< patch_files[pfile_map[pid]].name <<" from rank = " << mpi_handler::world_rank << " {" << 
            // sz_pref[0]<<","<<
            // sz_pref[1]<<","<<
            // sz_pref[2]<<","<<
            // sz_pref[3]<<","<<
            // sz_pref[4]<<","<<
            // sz_pref[5]<<"}"<<
            // std::endl;

            MPI_Status st;
            mpi::file_write(mfilepatch,sz_pref, 6, mpi_type_u32, &st);

            mpi::file_write(mfilepatch, pdat.pos_s.data(), sz_pref[0] , mpi_type_f32_3, &st);
            mpi::file_write(mfilepatch, pdat.pos_d.data(), sz_pref[1] , mpi_type_f64_3, &st);
            mpi::file_write(mfilepatch, pdat.U1_s.data() , sz_pref[2] , mpi_type_f32  , &st);
            mpi::file_write(mfilepatch, pdat.U1_d.data() , sz_pref[3] , mpi_type_f64  , &st);
            mpi::file_write(mfilepatch, pdat.U3_s.data() , sz_pref[4] , mpi_type_f32_3, &st);
            mpi::file_write(mfilepatch, pdat.U3_d.data() , sz_pref[5] , mpi_type_f64_3, &st);

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
            throw std::runtime_error("patch list size > u32_max not handled by dump");
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

        if(patchdata_layout::nVarpos_s == 1){
            u8 f = 0;
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.min_box_sim_s, 1, mpi_type_f32_3,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.max_box_sim_s, 1, mpi_type_f32_3,&st);
        }else{
            u8 f = 1;
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.min_box_sim_d, 1, mpi_type_f64_3,&st);    
            mpi::file_write(simbox_file, &sched.patch_data.sim_box.max_box_sim_d, 1, mpi_type_f64_3,&st);       
        }
        
    }



    mpi::file_close(&simbox_file);
}

inline void dump_state(std::string prefix, SchedulerMPI & sched){

    dump_patch_data(prefix, sched);

    dump_patch_list(prefix, sched);
    dump_simbox(prefix, sched);

}