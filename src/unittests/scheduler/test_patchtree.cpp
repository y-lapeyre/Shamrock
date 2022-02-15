#include "../../scheduler/patchtree.hpp"

#include "../shamrocktest.hpp"
#include <map>
#include <random>
#include <vector>
#include "../../scheduler/scheduler_patch_list.hpp"
#include "../../scheduler/hilbertsfc.hpp"


void recursprint(PatchTree &pt,std::vector<Patch>& plist,std::unordered_map<u64,u64> &idx_map, u64 idx, u32 indent){


    PTNode & ptnode = pt.tree[idx];


    for(u32 i = 0 ; i < indent; i++) std::cout << "   ";

    std::cout << "nodeid : " << idx << " ( " <<
                "[" << ptnode.x_min << "," << ptnode.x_max << "] " << 
                "[" << ptnode.y_min << "," << ptnode.y_max << "] " << 
                "[" << ptnode.z_min << "," << ptnode.z_max << "] " << 
                " ) ";


    if(ptnode.linked_patchid == u64_max){
        std::cout << std::endl;
        for(u8 child_id = 0; child_id < 8; child_id ++){
            recursprint(pt,plist,idx_map, ptnode.childs_id[child_id], indent +1);
        }
    }else{
        Patch & p = plist[idx_map[ptnode.linked_patchid]];
        std::cout << " node : " <<ptnode.linked_patchid<<" ( " <<
                "[" << p.x_min << "," << p.x_max << "] " << 
                "[" << p.y_min << "," << p.y_max << "] " << 
                "[" << p.z_min << "," << p.z_max << "] " << 
                " ) " << std::endl;
    }
    

}

Test_start("", testpatchtree, 1){

    std::vector<Patch> global = make_fake_patch_list(30,10);


    for(Patch & p : global){
        std::cout << p.id_patch << " ( " <<
            "[" << p.x_min << "," << p.x_max << "] " << 
            "[" << p.y_min << "," << p.y_max << "] " << 
            "[" << p.z_min << "," << p.z_max << "] " << 
            " ) " << "("<<p.x_max - p.x_min<<","<<p.y_max - p.y_min<<","<<p.z_max - p.z_min<<")" << std::endl;
    }

    std::cout << "number of ptch : " << global.size() << std::endl;

    PatchTree pt;

    pt.build_from_patchtable(global,hilbert_box21_sz);

    SchedulerPatchList plist;
    plist.global = global;

    plist.build_global_idx_map();

    recursprint(pt,global,plist.id_patch_to_global_idx, 0, 0);

}