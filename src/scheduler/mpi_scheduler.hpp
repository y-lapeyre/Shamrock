#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "patchdata.hpp"

class SchedulerPatchList{public:

    std::vector<Patch> global;
    std::vector<Patch> local;

    inline void sync_global(){
        mpi_handler::vector_allgatherv(local, patch_MPI_type, global, patch_MPI_type, MPI_COMM_WORLD);
    }

    inline std::unordered_set<u64> build_local(){

        std::unordered_set<u64> out_ids;

        local.clear();
        for(const Patch &p : global){
            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);
                out_ids.insert(p.id_patch);
            }
        }

        return out_ids;
        
    }

    

    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    inline void build_local_differantial(
        std::unordered_set<u64> & patch_id_lst, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx
        ){
        local.clear();

        for (u64 i = 0; i < global.size(); i++) {
            const Patch & p = global[i];

            bool was_owned = (patch_id_lst.find(p.id_patch) != patch_id_lst.end());

            //TODO add check node_owner_id valid 
            if(i32(p.node_owner_id) == mpi_handler::world_rank){
                local.push_back(p);

                if(!was_owned){
                    to_recv_idx.push_back(i);
                    patch_id_lst.insert(p.id_patch);
                }
            }else{
                if(was_owned){
                    to_send_idx.push_back(i);
                    patch_id_lst.erase(p.id_patch);
                }
            }
        }

    }


    /*
    inline void build_local_from_ids(const std::unordered_set<u64> & id_lst){
        local.clear();
        for(const Patch &p : global){
            if(id_lst.find(p.id_patch) != id_lst.end()){
                local.push_back(p);
            }
        }
    }
    */

};

class SchedulerPatchData{public:
    std::map<u64, PatchData> owned_data;

    inline void apply_change_list(std::vector<std::tuple<u32, i32, i32,i32>> change_list,SchedulerPatchList& patch_list){

        std::vector<MPI_Request> rq_lst;

        //send
        for(u32 i = 0 ; i < change_list.size(); i++){
            auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];

            //if i'm sender
            if(old_owner == mpi_handler::world_rank){
                auto & patchdata = owned_data[patch_list.global[idx].id_patch];
                patchdata_isend(patchdata, rq_lst, new_owner, tag_comm, MPI_COMM_WORLD);
            }
        }

        //receive
        for(u32 i = 0 ; i < change_list.size(); i++){
            auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];
            auto & id_patch = patch_list.global[idx].id_patch;
            
            //if i'm receiver
            if(new_owner == mpi_handler::world_rank){
                owned_data[id_patch] = patchdata_irecv( rq_lst, old_owner, tag_comm, MPI_COMM_WORLD);
            }
        }


        //wait
        std::vector<MPI_Status> st_lst(rq_lst.size());
        mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());


        //erase old patchdata
        for(u32 i = 0 ; i < change_list.size(); i++){
            auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];
            auto & id_patch = patch_list.global[idx].id_patch;
            
            patch_list.global[idx].node_owner_id = new_owner;

            //if i'm sender delete old data
            if(old_owner == mpi_handler::world_rank){
                owned_data.erase(id_patch);
            }

        }
    }
};



class MpiScheduler{public:


    SchedulerPatchList patch_list;
    SchedulerPatchData patch_data;

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id;

    



    inline void init_mpi_required_types(){
        if(!is_mpi_sycl_interop_active()){
            create_sycl_mpi_types();
        }

        if(!is_mpi_patch_type_active()){
            create_MPI_patch_type();
        }
    }

    inline void free_mpi_required_types(){
        if(is_mpi_sycl_interop_active()){
            free_sycl_mpi_types();
        }

        if(is_mpi_patch_type_active()){
            free_MPI_patch_type();
        }
    }

    inline MpiScheduler(){

    }

    inline virtual ~MpiScheduler(){

    }

};