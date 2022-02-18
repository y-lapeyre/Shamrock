#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include "../utils/geometry_utils.hpp"






struct PTNode{
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;
    u64 level;

    u64 parent_id;
    u64 childs_id[8] {u64_max};

    u64 linked_patchid = u64_max;

    bool is_leaf = true;
    bool child_are_all_leafs = false;

    //patch fields
    u64 data_count = u64_max;
    u64 load_value = u64_max;
};



class PatchTree{public:
    
    
    std::unordered_map<u64, PTNode> tree;

    std::unordered_set<u64> leaf_key;
    std::unordered_set<u64> parent_of_only_leaf_key;

    /**
     * @brief split a leaf node
     * 
     * @param id 
     */
    void split_node(u64 id);

    /**
     * @brief merge childs of idparent (id parent should have only leafs as childs)
     * 
     * @param idparent 
     */
    void merge_node_dm1(u64 idparent);

    /**
     * @brief make tree from a patch table
     * 
     * @param plist 
     * @param max_val_1axis 
     */
    void build_from_patchtable(std::vector<Patch> & plist, u64 max_val_1axis);

    /**
     * @brief update value in nodes (tree reduction) 
     * 
     * @param plist 
     * @param id_patch_to_global_idx 
     */
    void update_values_node(std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
    
    //TODO add function to run reduction on leafs and only_leafs_child node


    private:

    u64 next_id = 0;

    u64 insert_node(PTNode n);
    void remove_node(u64 id);

    void update_ptnode(PTNode & n,std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
};