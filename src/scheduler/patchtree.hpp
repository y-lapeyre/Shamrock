#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "../utils/geometry_utils.hpp"






struct PTNode{
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;

    u64 childs_id[8] {u64_max};

    u64 linked_patchid = u64_max;
};



class PatchTree{public:
    
    u64 next_id = 0;
    std::unordered_map<u64, PTNode> tree;

    u64 insert_node(PTNode n);
    
    void remove_node(u64 id);

    void split_node(u64 id);

    void merge_node(u64 idparent);



    


    void build_from_patchtable(std::vector<Patch> & plist, u64 max_val_1axis);


};