#include "scheduler_patch_list.hpp"
#include <vector>
#include <random>
#include "hilbertsfc.hpp"
#include "patch.hpp"

void SchedulerPatchList::sync_global(){
    mpi_handler::vector_allgatherv(local, patch_MPI_type, global, patch_MPI_type, MPI_COMM_WORLD);   
}


std::unordered_set<u64> SchedulerPatchList::build_local(){

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

void SchedulerPatchList::build_local_differantial(std::unordered_set<u64> &patch_id_lst, std::vector<u64> &to_send_idx, std::vector<u64> &to_recv_idx){
    
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




std::vector<Patch> make_fake_patch_list(u32 total_dtcnt,u64 div_limit){
    std::vector<Patch> plist;

    std::mt19937 eng(0x1111);        
    std::uniform_real_distribution<f32> split_val(0,1);     



    plist.push_back(Patch{
        0,
        u64_max,
        total_dtcnt,
        0,
        0,
        0,
        hilbert_box21_sz,
        hilbert_box21_sz,
        hilbert_box21_sz,
        total_dtcnt,
        0,
    });

    bool listchanged = true;

    u64 id_cnt = 0;
    while (listchanged){
        listchanged = false;


        std::vector<Patch> to_add;

        for(Patch & p : plist){
            if(p.data_count > div_limit){

                /*
                std::cout << "splitting : ( " <<
                    "[" << p.x_min << "," << p.x_max << "] " << 
                    "[" << p.y_min << "," << p.y_max << "] " << 
                    "[" << p.z_min << "," << p.z_max << "] " << 
                    " ) " << p.data_count <<  std::endl;
                    */
                
                u64 min_x = p.x_min;
                u64 min_y = p.y_min;
                u64 min_z = p.z_min;

                u64 split_x = (((p.x_max - p.x_min) + 1)/2) - 1 +min_x;
                u64 split_y = (((p.y_max - p.y_min) + 1)/2) - 1 +min_y;
                u64 split_z = (((p.z_max - p.z_min) + 1)/2) - 1 +min_z;

                u64 max_x = p.x_max;
                u64 max_y = p.y_max;
                u64 max_z = p.z_max;



                u32 qte_m = split_val(eng)*p.data_count;
                u32 qte_p = p.data_count - qte_m;

                u32 qte_mm = split_val(eng)*qte_m;
                u32 qte_mp = qte_m - qte_mm;

                u32 qte_pm = split_val(eng)*qte_p;
                u32 qte_pp = qte_p - qte_pm;


                u32 qte_mmm = split_val(eng)*qte_mm;
                u32 qte_mmp = qte_mm - qte_mmm;

                u32 qte_mpm = split_val(eng)*qte_mp;
                u32 qte_mpp = qte_mp - qte_mpm;

                u32 qte_pmm = split_val(eng)*qte_pm;
                u32 qte_pmp = qte_pm - qte_pmm;

                u32 qte_ppm = split_val(eng)*qte_pp;
                u32 qte_ppp = qte_pp - qte_ppm;



                Patch child_mmm = Patch{
                    id_cnt,
                    u64_max,
                    qte_mmm,
                    min_x,
                    min_y,
                    min_z,
                    split_x,
                    split_y,
                    split_z,
                    qte_mmm,
                    0,
                };id_cnt++;

                Patch child_mmp = Patch{
                    id_cnt,
                    u64_max,
                    qte_mmp,
                    min_x,
                    min_y,
                    split_z + 1,
                    split_x,
                    split_y,
                    max_z,
                    qte_mmp,
                    0,
                };id_cnt++;

                Patch child_mpm = Patch{
                    id_cnt,
                    u64_max,
                    qte_mpm,
                    min_x,
                    split_y+1,
                    min_z,
                    split_x,
                    max_y,
                    split_z,
                    qte_mpm,
                    0,
                };id_cnt++;

                Patch child_mpp = Patch{
                    id_cnt,
                    u64_max,
                    qte_mpp,
                    min_x,
                    split_y+1,
                    split_z+1,
                    split_x,
                    max_y,
                    max_z,
                    qte_mpp,
                    0,
                };id_cnt++;

                Patch child_pmm = Patch{
                    id_cnt,
                    u64_max,
                    qte_pmm,
                    split_x+1,
                    min_y,
                    min_z,
                    max_x,
                    split_y,
                    split_z,
                    qte_pmm,
                    0,
                };id_cnt++;

                Patch child_pmp = Patch{
                    id_cnt,
                    u64_max,
                    qte_pmp,
                    split_x+1,
                    min_y,
                    split_z+1,
                    max_x,
                    split_y,
                    max_z,
                    qte_pmp,
                    0,
                };id_cnt++;

                Patch child_ppm = Patch{
                    id_cnt,
                    u64_max,
                    qte_ppm,
                    split_x+1,
                    split_y+1,
                    min_z,
                    max_x,
                    max_y,
                    split_z,
                    qte_ppm,
                    0,
                };id_cnt++;

                Patch child_ppp = Patch{
                    id_cnt,
                    u64_max,
                    qte_ppp,
                    split_x+1,
                    split_y+1,
                    split_z+1,
                    max_x,
                    max_y,
                    max_z,
                    qte_ppp,
                    0,
                };id_cnt++;



                
                p = child_mmm;
                to_add.push_back(child_mmp);
                to_add.push_back(child_mpm);
                to_add.push_back(child_mpp);
                to_add.push_back(child_pmm);
                to_add.push_back(child_pmp);
                to_add.push_back(child_ppm);
                to_add.push_back(child_ppp);
                

                
            }
        }

        if(! to_add.empty()){
            listchanged = true;

            plist.insert(plist.end(),to_add.begin(),to_add.end());
        }

        /*
        for(Patch & p : plist){
            std::cout << "( " <<
                "[" << p.x_min << "," << p.x_max << "] " << 
                "[" << p.y_min << "," << p.y_max << "] " << 
                "[" << p.z_min << "," << p.z_max << "] " << 
                " ) " << p.data_count << std::endl;
        }

        std::cout << "----- end cycle -----" << std::endl;
        */

    }

    



    return plist;
}