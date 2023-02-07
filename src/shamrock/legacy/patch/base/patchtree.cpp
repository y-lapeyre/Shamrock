// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchtree.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief PatchTree implementation
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "patchtree.hpp"

#include <stdexcept>
#include <vector>
#include "shamrock/patch/Patch.hpp"


u64 PatchTree::insert_node(PTNode n){
    tree[next_id] = n;
    next_id ++;
    return next_id-1;
}

void PatchTree::remove_node(u64 id){
    tree.erase(id);
}

void PatchTree::split_node(u64 id){

    leaf_key.erase(id);

    if(tree[id].parent_id != u64_max){
        parent_of_only_leaf_key.erase(tree[id].parent_id);
        tree[tree[id].parent_id].child_are_all_leafs = false;
    }

    PTNode& curr = tree[id];
    curr.is_leaf = false;
    curr.child_are_all_leafs = true;

    u64 min_x = curr.x_min;
    u64 min_y = curr.y_min;
    u64 min_z = curr.z_min;

    u64 split_x = (((curr.x_max - curr.x_min) + 1)/2) - 1 + min_x;
    u64 split_y = (((curr.y_max - curr.y_min) + 1)/2) - 1 + min_y;
    u64 split_z = (((curr.z_max - curr.z_min) + 1)/2) - 1 + min_z;

    u64 max_x = curr.x_max;
    u64 max_y = curr.y_max;
    u64 max_z = curr.z_max;

    curr.childs_id[0] = insert_node(PTNode{
        min_x,
        min_y,
        min_z,
        split_x,
        split_y,
        split_z,
        (curr.level +1),
        id
    });

    curr.childs_id[1] = insert_node(PTNode{
        min_x,
        min_y,
        split_z + 1,
        split_x,
        split_y,
        max_z,
        (curr.level +1),
        id
    });

    curr.childs_id[2] = insert_node(PTNode{
        min_x,
        split_y+1,
        min_z,
        split_x,
        max_y,
        split_z,
        (curr.level +1),
        id
    });

    curr.childs_id[3] = insert_node(PTNode{
        min_x,
        split_y+1,
        split_z+1,
        split_x,
        max_y,
        max_z,
        (curr.level +1),
        id
    });

    curr.childs_id[4] = insert_node(PTNode{
        split_x+1,
        min_y,
        min_z,
        max_x,
        split_y,
        split_z,
        (curr.level +1),
        id
    });

    curr.childs_id[5] = insert_node(PTNode{
        split_x+1,
        min_y,
        split_z+1,
        max_x,
        split_y,
        max_z,
        (curr.level +1),
        id
    });

    curr.childs_id[6] = insert_node(PTNode{
        split_x+1,
        split_y+1,
        min_z,
        max_x,
        max_y,
        split_z,
        (curr.level +1),
        id
    });

    curr.childs_id[7] = insert_node(PTNode{
        split_x+1,
        split_y+1,
        split_z+1,
        max_x,
        max_y,
        max_z,
        (curr.level +1),
        id
    });

    parent_of_only_leaf_key.insert(id);
    leaf_key.insert(curr.childs_id[0]);
    leaf_key.insert(curr.childs_id[1]);
    leaf_key.insert(curr.childs_id[2]);
    leaf_key.insert(curr.childs_id[3]);
    leaf_key.insert(curr.childs_id[4]);
    leaf_key.insert(curr.childs_id[5]);
    leaf_key.insert(curr.childs_id[6]);
    leaf_key.insert(curr.childs_id[7]);

}

void PatchTree::merge_node_dm1(u64 idparent){

    if(!tree[idparent].child_are_all_leafs ){
        throw shamrock_exc("node should be parent of only leafs");
    }
    
    leaf_key.erase(tree[idparent].childs_id[0]);
    leaf_key.erase(tree[idparent].childs_id[1]);
    leaf_key.erase(tree[idparent].childs_id[2]);
    leaf_key.erase(tree[idparent].childs_id[3]);
    leaf_key.erase(tree[idparent].childs_id[4]);
    leaf_key.erase(tree[idparent].childs_id[5]);
    leaf_key.erase(tree[idparent].childs_id[6]);
    leaf_key.erase(tree[idparent].childs_id[7]);

    remove_node(tree[idparent].childs_id[0]);
    remove_node(tree[idparent].childs_id[1]);
    remove_node(tree[idparent].childs_id[2]);
    remove_node(tree[idparent].childs_id[3]);
    remove_node(tree[idparent].childs_id[4]);
    remove_node(tree[idparent].childs_id[5]);
    remove_node(tree[idparent].childs_id[6]);
    remove_node(tree[idparent].childs_id[7]);

    tree[idparent].childs_id[0] = u64_max;
    tree[idparent].childs_id[1] = u64_max;
    tree[idparent].childs_id[2] = u64_max;
    tree[idparent].childs_id[3] = u64_max;
    tree[idparent].childs_id[4] = u64_max;
    tree[idparent].childs_id[5] = u64_max;
    tree[idparent].childs_id[6] = u64_max;
    tree[idparent].childs_id[7] = u64_max;

    leaf_key.insert(idparent);
    tree[idparent].is_leaf = true;

    parent_of_only_leaf_key.erase(idparent);
    tree[idparent].child_are_all_leafs = false;
    
    //check if parent of tree[idparent] is parent of only leafs
    if(tree[idparent].parent_id != u64_max){
        bool only_leafs = true;

        PTNode & parent_node = tree[tree[idparent].parent_id];

        for(u8 idc = 0; idc < 8 ; idc ++){
            only_leafs = only_leafs && tree[parent_node.childs_id[idc]].is_leaf;
        }

        if(only_leafs){
            parent_node.child_are_all_leafs = true;
            parent_of_only_leaf_key.insert(tree[idparent].parent_id);
        }
    }
    
}






void PatchTree::build_from_patchtable(std::vector<shamrock::patch::Patch> & plist, u64 max_val_1axis){

    if(plist.size() > 1){
        PTNode root;
        root.x_max = max_val_1axis;
        root.y_max = max_val_1axis;
        root.z_max = max_val_1axis;
        root.x_min = 0;
        root.y_min = 0;
        root.z_min = 0;
        root.level = 0;
        root.parent_id = u64_max;

        u64 root_id = insert_node(root);
        leaf_key.insert(root_id);
        roots_id.insert(root_id);



        std::vector<u64> complete_vec;
        for(u64 i = 0; i < plist.size(); i++){
            complete_vec.push_back(i);
        }

        std::vector< std::tuple<u64,std::vector<u64>> > tree_vec(1);

        tree_vec[0] = {root_id,complete_vec};

        while(tree_vec.size()>0){
            std::vector< std::tuple<u64,std::vector<u64>> > next_tree_vec;
            for(auto & [idtree,idvec] : tree_vec){

                PTNode & ptn = tree[idtree];

                split_node(idtree);



                for(u8 child_id = 0; child_id < 8; child_id ++){

                    u64 ptnode_id = ptn.childs_id[child_id];
                    std::vector<u64> buf;

                    PTNode & curr = tree[ptnode_id];

                    for(u64 idxptch : idvec){
                        shamrock::patch::Patch &p = plist[idxptch];

                        bool is_inside = BBAA::iscellb_inside_a<u32_3>({curr.x_min,curr.y_min,curr.z_min},{curr.x_max,curr.y_max,curr.z_max},
                            {p.x_min,p.y_min,p.z_min},{p.x_max,p.y_max,p.z_max});

                        if(is_inside){buf.push_back(idxptch); }

                        /*
                        std::cout << " ( " <<
                            "[" << curr.x_min << "," << curr.x_max << "] " << 
                            "[" << curr.y_min << "," << curr.y_max << "] " << 
                            "[" << curr.z_min << "," << curr.z_max << "] " << 
                            " )  node : ( " <<
                            "[" << p.x_min << "," << p.x_max << "] " << 
                            "[" << p.y_min << "," << p.y_max << "] " << 
                            "[" << p.z_min << "," << p.z_max << "] " << 
                            " ) "<< is_inside;

                        if(is_inside){ std::cout << " -> push " << idxptch;}

                        std::cout << std::endl;
                        */
                    }

                    if(buf.size() == 1){
                        //std::cout << "set linked id node " << buf[0] << " : "  << plist[buf[0]].id_patch << std::endl;
                        tree[ptnode_id].linked_patchid = plist[buf[0]].id_patch;
                    }else{
                        next_tree_vec.push_back({ptnode_id,buf});
                    }

                }

            }//std::cout << "----------------" << std::endl;

            tree_vec = next_tree_vec;
        }
    }else if(plist.size() == 1){

        PTNode root;
        root.x_max = max_val_1axis;
        root.y_max = max_val_1axis;
        root.z_max = max_val_1axis;
        root.x_min = 0;
        root.y_min = 0;
        root.z_min = 0;
        root.level = 0;
        root.parent_id = u64_max;
        root.is_leaf = true;
        root.child_are_all_leafs = false;
        root.linked_patchid = plist[0].id_patch;

        u64 root_id = insert_node(root);
        leaf_key.insert(root_id);
        roots_id.insert(root_id);

    }


}

void PatchTree::update_ptnode(PTNode & n,std::vector<shamrock::patch::Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx){

    if(n.linked_patchid != u64_max){
        n.data_count = 0;
        n.load_value = 0;
        n.data_count += plist[id_patch_to_global_idx[n.linked_patchid]].data_count;
        n.load_value += plist[id_patch_to_global_idx[n.linked_patchid]].load_value;
    }else if (n.childs_id[0] != u64_max) {

        bool has_err_val = false;

        n.data_count = 0;
        n.load_value = 0;
        for(u8 idc = 0; idc < 8 ; idc ++){

            if(tree[n.childs_id[idc]].data_count == u64_max)has_err_val = true;

            n.data_count += tree[n.childs_id[idc]].data_count;
            n.load_value += tree[n.childs_id[idc]].load_value;
        }
        
        if(has_err_val){
            n.data_count = u64_max;
            n.load_value = u64_max;
        }
        
    }

}

// TODO add test value on root = sum all leaf
void PatchTree::update_values_node(std::vector<shamrock::patch::Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx){


    tree[0].data_count = u64_max;

    while(tree[0].data_count == u64_max){
        for(auto & [key,ptnode] : tree){
            update_ptnode(ptnode,plist,id_patch_to_global_idx);
        }
    }

}

void PatchTree::partial_values_reduction(std::vector<shamrock::patch::Patch> &plist, std::unordered_map<u64, u64> id_patch_to_global_idx){

    for( u64 id_leaf : leaf_key){
        update_ptnode(tree[id_leaf],plist,id_patch_to_global_idx);
    }

    for( u64 id_leaf : parent_of_only_leaf_key){
        update_ptnode(tree[id_leaf],plist,id_patch_to_global_idx);
    }

}



std::unordered_set<u64> PatchTree::get_split_request(u64 crit_load_split){

    std::unordered_set<u64> rq;

    for(u64 a : leaf_key){
        if (tree[a].load_value > crit_load_split) {
            rq.insert(a);
        }
    }

    return rq;

}


std::unordered_set<u64> PatchTree::get_merge_request(u64 crit_load_merge){
    std::unordered_set<u64> rq;

    for(u64 a : parent_of_only_leaf_key){
        if (tree[a].load_value < crit_load_merge) {
            rq.insert(a);
        }
    }

    return rq;
}



