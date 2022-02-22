#include "scheduler_patch_data.hpp"
#include "patchdata.hpp"
#include <vector>

void SchedulerPatchData::apply_change_list(std::vector<std::tuple<u64, i32, i32,i32>> change_list,SchedulerPatchList& patch_list){

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




bool inside_crit();



void SchedulerPatchData::split_patchdata_pos_s(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7){

    PatchData & original_pd = owned_data[key_orginal];

    if(patchdata_layout::nVarpos_s == 0){

    }

    std::vector<u64> attrib_map = std::vector<u64>();

}

void SchedulerPatchData::split_patchdata_pos_d(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7){

    

}


void SchedulerPatchData::split_patchdata(u64 key_orginal,Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7){

    if(patchdata_layout::nVarpos_s == 1)split_patchdata_pos_s(key_orginal, p0, p1, p2, p3, p4, p5, p6, p7);

    if(patchdata_layout::nVarpos_d == 1)split_patchdata_pos_d(key_orginal, p0, p1, p2, p3, p4, p5, p6, p7);

}
