#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <cstdio>
#include <unordered_map>
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

    

    //patch fields
    u64 data_count = u64_max;
    u64 load_value = u64_max;
};



class PatchTree{public:
    
    
    std::unordered_map<u64, PTNode> tree;

    

    void split_node(u64 id);
    void merge_node(u64 idparent);


    void build_from_patchtable(std::vector<Patch> & plist, u64 max_val_1axis);

    void update_values_node(std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
    


    private:

    u64 next_id = 0;

    u64 insert_node(PTNode n);
    void remove_node(u64 id);

    void update_ptnode(PTNode & n,std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
};