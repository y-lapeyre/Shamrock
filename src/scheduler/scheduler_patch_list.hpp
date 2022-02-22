#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "patch.hpp"

class SchedulerPatchList{public:

    u64 _next_patch_id = 0;

    std::vector<Patch> global;
    std::vector<Patch> local;




    void sync_global();




    std::unordered_set<u64> build_local();

    
    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    void build_local_differantial(
        std::unordered_set<u64> & patch_id_lst, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx
        );


    std::unordered_map<u64,u64> id_patch_to_global_idx;
    inline void build_global_idx_map(){
        id_patch_to_global_idx.clear();

        u64 idx = 0;
        for(Patch p : global){
            id_patch_to_global_idx[p.id_patch]  = idx;
            idx ++;
        }

    }


    std::unordered_map<u64,u64> id_patch_to_local_idx;
    inline void build_local_idx_map(){
        id_patch_to_local_idx.clear();

        u64 idx = 0;
        for(Patch p : local){
            id_patch_to_local_idx[p.id_patch]  = idx;
            idx ++;
        }

    }


    inline std::tuple<u64,u64,u64,u64,u64,u64,u64,u64> split_patch(u64 id_patch){

        Patch & p0 = global[id_patch_to_global_idx[id_patch]];

        Patch p1,p2,p3,p4,p5,p6,p7;

        split_patch_obj(p0, p1, p2, p3, p4, p5, p6, p7);
        
        p1.id_patch = _next_patch_id;
        _next_patch_id ++;

        p2.id_patch = _next_patch_id;
        _next_patch_id ++;

        p3.id_patch = _next_patch_id;
        _next_patch_id ++;

        p4.id_patch = _next_patch_id;
        _next_patch_id ++;

        p5.id_patch = _next_patch_id;
        _next_patch_id ++;

        p6.id_patch = _next_patch_id;
        _next_patch_id ++;

        p7.id_patch = _next_patch_id;
        _next_patch_id ++;

        u64 idx_p1 = global.size();
        global.push_back(p1);

        u64 idx_p2 = idx_p1 +1 ;
        global.push_back(p2);

        u64 idx_p3 = idx_p2 +1 ;
        global.push_back(p3);

        u64 idx_p4 = idx_p3 +1 ;
        global.push_back(p4);

        u64 idx_p5 = idx_p4 +1 ;
        global.push_back(p5);

        u64 idx_p6 = idx_p5 +1 ;
        global.push_back(p6);

        u64 idx_p7 = idx_p6 +1 ;
        global.push_back(p7);

        return {id_patch_to_global_idx[id_patch],
                idx_p1,idx_p2,idx_p3,idx_p4,idx_p5,idx_p6,idx_p7
            };

    }
    

};



std::vector<Patch> make_fake_patch_list(u32 total_dtcnt,u64 div_limit);