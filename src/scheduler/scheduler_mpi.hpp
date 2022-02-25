#pragma once

#include "patch.hpp"
#include "../sys/mpi_handler.hpp"
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "patchdata.hpp"

#include "patchtree.hpp"
#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"



class SchedulerMPI{public:

    u64 crit_patch_split;
    u64 crit_patch_merge;


    SchedulerPatchList patch_list;
    SchedulerPatchData patch_data;
    PatchTree patch_tree;

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id;


    


    void sync_build_LB(bool global_patch_sync, bool balance_load);

    void scheduler_step(bool do_split_merge,bool do_load_balancing);
    
    inline void add_patch(Patch & p, PatchData & pdat){
        p.id_patch = patch_list._next_patch_id;
        patch_list._next_patch_id ++;

        patch_list.global.push_back(p);

        patch_data.owned_data[p.id_patch] = pdat;

        
    }

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

    inline SchedulerMPI(u64 crit_split,u64 crit_merge){

        crit_patch_split = crit_split;
        crit_patch_merge = crit_merge;
        

    }

    inline virtual ~SchedulerMPI(){

    }



    std::string dump_status();


    inline void update_local_dtcnt_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].data_count = patch_data.owned_data[id].pos_s.size() + patch_data.owned_data[id].pos_d.size() ;
        }
    }

    inline void update_local_load_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].load_value = patch_data.owned_data[id].pos_s.size() + patch_data.owned_data[id].pos_d.size() ;
        }
    }

    private:


    
    inline void split_patches(std::unordered_set<u64> split_rq){
        for(u64 tree_id : split_rq){

            patch_tree.split_node(tree_id);
            PTNode & splitted_node = patch_tree.tree[tree_id];

            auto [idx_p0,idx_p1,idx_p2,idx_p3,idx_p4,idx_p5,idx_p6,idx_p7] 
                =  patch_list.split_patch(splitted_node.linked_patchid);

            u64 old_patch_id = splitted_node.linked_patchid;

            splitted_node.linked_patchid = u64_max;
            patch_tree.tree[splitted_node.childs_id[0]].linked_patchid = patch_list.global[idx_p0].id_patch;
            patch_tree.tree[splitted_node.childs_id[1]].linked_patchid = patch_list.global[idx_p1].id_patch;
            patch_tree.tree[splitted_node.childs_id[2]].linked_patchid = patch_list.global[idx_p2].id_patch;
            patch_tree.tree[splitted_node.childs_id[3]].linked_patchid = patch_list.global[idx_p3].id_patch;
            patch_tree.tree[splitted_node.childs_id[4]].linked_patchid = patch_list.global[idx_p4].id_patch;
            patch_tree.tree[splitted_node.childs_id[5]].linked_patchid = patch_list.global[idx_p5].id_patch;
            patch_tree.tree[splitted_node.childs_id[6]].linked_patchid = patch_list.global[idx_p6].id_patch;
            patch_tree.tree[splitted_node.childs_id[7]].linked_patchid = patch_list.global[idx_p7].id_patch;

            patch_data.split_patchdata(
                old_patch_id,
                patch_list.global[idx_p0], 
                patch_list.global[idx_p1],
                patch_list.global[idx_p2],
                patch_list.global[idx_p3],
                patch_list.global[idx_p4],
                patch_list.global[idx_p5],
                patch_list.global[idx_p6],
                patch_list.global[idx_p7]);

        }
    }

    inline void merge_patches(std::unordered_set<u64> merge_rq){
        for(u64 tree_id : merge_rq){

            PTNode & to_merge_node = patch_tree.tree[tree_id];

            std::cout << "merging patch tree id : " << tree_id << "\n";
            

            u64 patch_id0 = patch_tree.tree[to_merge_node.childs_id[0]].linked_patchid;
            u64 patch_id1 = patch_tree.tree[to_merge_node.childs_id[1]].linked_patchid;
            u64 patch_id2 = patch_tree.tree[to_merge_node.childs_id[2]].linked_patchid;
            u64 patch_id3 = patch_tree.tree[to_merge_node.childs_id[3]].linked_patchid;
            u64 patch_id4 = patch_tree.tree[to_merge_node.childs_id[4]].linked_patchid;
            u64 patch_id5 = patch_tree.tree[to_merge_node.childs_id[5]].linked_patchid;
            u64 patch_id6 = patch_tree.tree[to_merge_node.childs_id[6]].linked_patchid;
            u64 patch_id7 = patch_tree.tree[to_merge_node.childs_id[7]].linked_patchid;
            
            
            std::cout << format("  -> (%d %d %d %d %d %d %d %d)\n", patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);
            
            if(patch_list.global[patch_list.id_patch_to_global_idx[ patch_id0 ]].node_owner_id == mpi_handler::world_rank){
                patch_data.merge_patchdata(patch_id0, patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);
            }

            patch_list.merge_patch(
                patch_list.id_patch_to_global_idx[ patch_id0 ],
                patch_list.id_patch_to_global_idx[ patch_id1 ],
                patch_list.id_patch_to_global_idx[ patch_id2 ],
                patch_list.id_patch_to_global_idx[ patch_id3 ],
                patch_list.id_patch_to_global_idx[ patch_id4 ],
                patch_list.id_patch_to_global_idx[ patch_id5 ],
                patch_list.id_patch_to_global_idx[ patch_id6 ],
                patch_list.id_patch_to_global_idx[ patch_id7 ]);

            patch_tree.merge_node_dm1(tree_id);

            to_merge_node.linked_patchid = patch_id0;

        }
    }


    inline void set_patch_pack_values(std::unordered_set<u64> merge_rq){

        for(u64 tree_id : merge_rq){

            PTNode & to_merge_node = patch_tree.tree[tree_id];

            u64 idx_pack = patch_list.id_patch_to_global_idx[
                patch_tree.tree[to_merge_node.childs_id[0]].linked_patchid
                ];

            //std::cout << "node id : " << patch_list.global[idx_pack].id_patch << " should merge with : ";

            for (u8 i = 1; i < 8; i++) {
                //std::cout <<  patch_tree.tree[to_merge_node.childs_id[i]].linked_patchid << " ";
                patch_list.global[
                    patch_list.id_patch_to_global_idx[
                            patch_tree.tree[to_merge_node.childs_id[i]].linked_patchid
                        ]
                    ].pack_node_index = idx_pack;
            }//std::cout << std::endl;

        }

    }

   


};