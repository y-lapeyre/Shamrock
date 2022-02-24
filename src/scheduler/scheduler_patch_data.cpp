#include "scheduler_patch_data.hpp"
#include "hilbertsfc.hpp"
#include "patchdata.hpp"
#include <stdexcept>
#include <vector>

#include "../utils/geometry_utils.hpp"

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




template<class Vectype>
void split_patchdata(
    PatchData & original_pd,
    const Vectype & min_box_sim,const Vectype & max_box_sim,
    Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7,
    PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData & pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7);















//TODO recode with better parralelism
//TODO hilbert_box21_sz shouldn't be here
//TODO refactor the SchedulerMPI with templated space filling curve to avoid using hilbert_box21_sz

template<>
void split_patchdata<f32_3>(PatchData & original_pd,
    const f32_3 & min_box_sim,const f32_3 & max_box_sim,
    Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7,
    PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData & pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7){

    f32_3 translate_factor = min_box_sim;
    f32_3 scale_factor = (max_box_sim - min_box_sim)/hilbert_box21_sz;

    f32_3 bmin_p0 = f32_3{p0.x_min,p0.y_min,p0.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p1 = f32_3{p1.x_min,p1.y_min,p1.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p2 = f32_3{p2.x_min,p2.y_min,p2.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p3 = f32_3{p3.x_min,p3.y_min,p3.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p4 = f32_3{p4.x_min,p4.y_min,p4.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p5 = f32_3{p5.x_min,p5.y_min,p5.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p6 = f32_3{p6.x_min,p6.y_min,p6.z_min}*scale_factor + translate_factor;
    f32_3 bmin_p7 = f32_3{p7.x_min,p7.y_min,p7.z_min}*scale_factor + translate_factor;

    f32_3 bmax_p0 = (f32_3{p0.x_max,p0.y_max,p0.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p1 = (f32_3{p1.x_max,p1.y_max,p1.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p2 = (f32_3{p2.x_max,p2.y_max,p2.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p3 = (f32_3{p3.x_max,p3.y_max,p3.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p4 = (f32_3{p4.x_max,p4.y_max,p4.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p5 = (f32_3{p5.x_max,p5.y_max,p5.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p6 = (f32_3{p6.x_max,p6.y_max,p6.z_max}+ 1)*scale_factor + translate_factor;
    f32_3 bmax_p7 = (f32_3{p7.x_max,p7.y_max,p7.z_max}+ 1)*scale_factor + translate_factor;

    for (u32 i = 0; i < original_pd.pos_s.size(); i++) {
        
        bool bp0 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p0,bmax_p0);
        bool bp1 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p1,bmax_p1);
        bool bp2 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p2,bmax_p2);
        bool bp3 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p3,bmax_p3);
        bool bp4 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p4,bmax_p4);
        bool bp5 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p5,bmax_p5);
        bool bp6 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p6,bmax_p6);
        bool bp7 = BBAA::is_particle_in_patch<f32_3>( original_pd.pos_s[i], bmin_p7,bmax_p7);

        if(bp0){
            pd0.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd0.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp1){
            pd1.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd1.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp2){
            pd2.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd2.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp3){
            pd3.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd3.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp4){
            pd4.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd4.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp5){
            pd5.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd5.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp6){
            pd6.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd6.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp7){
            pd7.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd7.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }


    }

}

template<>
void split_patchdata<f64_3>(PatchData & original_pd,
    const f64_3 & min_box_sim,const f64_3 & max_box_sim,
    Patch & p0,Patch & p1,Patch & p2,Patch & p3,Patch & p4,Patch & p5,Patch & p6,Patch & p7,
    PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData & pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7){


    f64_3 translate_factor = min_box_sim;
    f64_3 scale_factor = max_box_sim - min_box_sim;

    f64_3 bmin_p0 = f64_3{p0.x_min,p0.y_min,p0.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p1 = f64_3{p1.x_min,p1.y_min,p1.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p2 = f64_3{p2.x_min,p2.y_min,p2.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p3 = f64_3{p3.x_min,p3.y_min,p3.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p4 = f64_3{p4.x_min,p4.y_min,p4.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p5 = f64_3{p5.x_min,p5.y_min,p5.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p6 = f64_3{p6.x_min,p6.y_min,p6.z_min}*scale_factor + translate_factor;
    f64_3 bmin_p7 = f64_3{p7.x_min,p7.y_min,p7.z_min}*scale_factor + translate_factor;

    f64_3 bmax_p0 = (f64_3{p0.x_max,p0.y_max,p0.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p1 = (f64_3{p1.x_max,p1.y_max,p1.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p2 = (f64_3{p2.x_max,p2.y_max,p2.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p3 = (f64_3{p3.x_max,p3.y_max,p3.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p4 = (f64_3{p4.x_max,p4.y_max,p4.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p5 = (f64_3{p5.x_max,p5.y_max,p5.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p6 = (f64_3{p6.x_max,p6.y_max,p6.z_max}+ 1)*scale_factor + translate_factor;
    f64_3 bmax_p7 = (f64_3{p7.x_max,p7.y_max,p7.z_max}+ 1)*scale_factor + translate_factor;

    for (u32 i = 0; i < original_pd.pos_s.size(); i++) {
        
        bool bp0 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p0,bmax_p0);
        bool bp1 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p1,bmax_p1);
        bool bp2 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p2,bmax_p2);
        bool bp3 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p3,bmax_p3);
        bool bp4 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p4,bmax_p4);
        bool bp5 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p5,bmax_p5);
        bool bp6 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p6,bmax_p6);
        bool bp7 = BBAA::is_particle_in_patch<f64_3>( original_pd.pos_d[i], bmin_p7,bmax_p7);

        if(bp0){
            pd0.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd0.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd0.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp1){
            pd1.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd1.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd1.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp2){
            pd2.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd2.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd2.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp3){
            pd3.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd3.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd3.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp4){
            pd4.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd4.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd4.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp5){
            pd5.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd5.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd5.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp6){
            pd6.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd6.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd6.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }

        if(bp7){
            pd7.pos_s.push_back(original_pd.pos_s[i]);
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U1_s.push_back(original_pd.U1_s[i*patchdata_layout::nVarU1_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U1_d.push_back(original_pd.U1_d[i*patchdata_layout::nVarU1_d + j]);
            }
            
            for(u32 j = 0 ; j < patchdata_layout::nVarU1_s ; j++){
                pd7.U3_s.push_back(original_pd.U3_s[i*patchdata_layout::nVarU3_s + j]);
            }
            for(u32 j = 0 ; j < patchdata_layout::nVarU3_s ; j++){
                pd7.U3_d.push_back(original_pd.U3_d[i*patchdata_layout::nVarU3_d + j]);
            }
        }


    }

}


















void SchedulerPatchData::split_patchdata(u64 key_orginal, Patch &p0, Patch &p1, Patch &p2, Patch &p3, Patch &p4, Patch &p5, Patch &p6, Patch &p7){

    //TODO maybe unecessary
    if(patchdata_layout::nVarpos_s + patchdata_layout::nVarpos_d != 1) 
        throw std::runtime_error("nVarpos_s + nVarpos_d should be equal to 1");
    
    auto search = owned_data.find(key_orginal);

    if (search != owned_data.end()) {

        PatchData & original_pd = search->second;

        PatchData pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7;

        if(patchdata_layout::nVarpos_s == 1) {
            ::split_patchdata<f32_3>(
                    original_pd,
                    sim_box.min_box_sim_s,sim_box.max_box_sim_s,
                    p0,p1,p2,p3,p4,p5,p6,p7,
                    pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7);
        }

        if(patchdata_layout::nVarpos_d == 1) {
            ::split_patchdata<f64_3>(
                    original_pd,
                    sim_box.min_box_sim_d,sim_box.max_box_sim_d,
                    p0,p1,p2,p3,p4,p5,p6,p7,
                    pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7);
        }

        owned_data.erase(key_orginal);

        owned_data[p0.id_patch] = pd0;
        owned_data[p1.id_patch] = pd1;
        owned_data[p2.id_patch] = pd2;
        owned_data[p3.id_patch] = pd3;
        owned_data[p4.id_patch] = pd4;
        owned_data[p5.id_patch] = pd5;
        owned_data[p6.id_patch] = pd6;
        owned_data[p7.id_patch] = pd7;
    }

}




