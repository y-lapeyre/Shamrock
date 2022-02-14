#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <cstdio>
#include <unordered_map>
#include <vector>




inline bool cell_intersection_nonnull(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb){
    // printf("x : [%f %f] [%f %f]\n", pos_min_cella.x() ,pos_max_cella.x(), pos_min_cellb.x(),pos_max_cellb.x());
    // printf("y : [%f %f] [%f %f]\n", pos_min_cella.y(), pos_max_cella.y() ,pos_min_cellb.y(),pos_max_cellb.y());
    // printf("z : [%f %f] [%f %f]\n", pos_min_cella.z(), pos_max_cella.z() ,pos_min_cellb.z(),pos_max_cellb.z());

    return (
            (sycl::max( pos_min_cella.x(), pos_min_cellb.x()) < sycl::min(pos_max_cella.x(),pos_max_cellb.x())) &&
            (sycl::max( pos_min_cella.y(), pos_min_cellb.y()) < sycl::min(pos_max_cella.y(),pos_max_cellb.y())) &&
            (sycl::max( pos_min_cella.z(), pos_min_cellb.z()) < sycl::min(pos_max_cella.z(),pos_max_cellb.z())) 
        );
}




struct PTNode{
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;

    u64 childs_id[8] {u64_max};
};

class PatchTree{public:
    
    u64 next_id = 0;
    std::unordered_map<u64, PTNode> tree;

    inline u64 insert_node(PTNode && n){
        tree[next_id] = n;
        next_id ++;
        return next_id-1;
    }
    
    inline void remove_node(u64 id){
        tree.erase(id);
    }

    inline void split_node(u64 id){
        PTNode& curr = tree[id];

        u64 min_x = curr.x_min;
        u64 min_y = curr.y_min;
        u64 min_z = curr.z_min;

        u64 split_x = (((curr.x_max - curr.x_min) + 1)/2) - 1 ;
        u64 split_y = (((curr.y_max - curr.y_min) + 1)/2) - 1 ;
        u64 split_z = (((curr.z_max - curr.z_min) + 1)/2) - 1 ;

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
        });

        curr.childs_id[1] = insert_node(PTNode{
            min_x,
            min_y,
            split_z + 1,
            split_x,
            split_y,
            max_z,
        });

        curr.childs_id[2] = insert_node(PTNode{
            min_x,
            split_y+1,
            min_z,
            split_x,
            max_y,
            split_z,
        });

        curr.childs_id[3] = insert_node(PTNode{
            min_x,
            split_y+1,
            split_z+1,
            split_x,
            max_y,
            max_z,
        });

        curr.childs_id[4] = insert_node(PTNode{
            split_x+1,
            min_y,
            min_z,
            max_x,
            split_y,
            split_z,
        });

        curr.childs_id[5] = insert_node(PTNode{
            split_x+1,
            min_y,
            split_z+1,
            max_x,
            split_y,
            max_z,
        });

        curr.childs_id[6] = insert_node(PTNode{
            split_x+1,
            split_y+1,
            min_z,
            max_x,
            max_y,
            split_z,
        });

        curr.childs_id[7] = insert_node(PTNode{
            split_x+1,
            split_y+1,
            split_z+1,
            max_x,
            max_y,
            max_z,
        });

    }

    inline void merge_node(u64 idparent){

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
    }




    inline void insert_patch(Patch&p){

        

    }


    inline void build_from_patchtable(std::vector<Patch> & p){



    }


};