#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <cstdio>
#include <unordered_map>
#include <vector>




inline bool iscellb_inside_a(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb){

    return (
            (pos_min_cella.x() <= pos_min_cellb.x()) && (pos_min_cellb.x() < pos_max_cellb.x()) && (pos_max_cellb.x() <= pos_max_cella.x()) &&
            (pos_min_cella.y() <= pos_min_cellb.y()) && (pos_min_cellb.y() < pos_max_cellb.y()) && (pos_max_cellb.y() <= pos_max_cella.y()) &&
            (pos_min_cella.z() <= pos_min_cellb.z()) && (pos_min_cellb.z() < pos_max_cellb.z()) && (pos_max_cellb.z() <= pos_max_cella.z()) 
        );
        
}




struct PTNode{
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;

    u64 childs_id[8] {u64_max};

    u64 linked_patchid = u64_max;
};



class PatchTree{public:
    
    u64 next_id = 0;
    std::unordered_map<u64, PTNode> tree;

    inline u64 insert_node(PTNode n){
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



    


    inline void build_from_patchtable(std::vector<Patch> & plist, u64 max_val_1axis){


        PTNode root;
        root.x_max = max_val_1axis;
        root.y_max = max_val_1axis;
        root.z_max = max_val_1axis;
        root.x_min = 0;
        root.y_min = 0;
        root.z_min = 0;

        u64 root_id = insert_node(root);



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
                        Patch &p = plist[idxptch];

                        bool is_inside = iscellb_inside_a({curr.x_min,curr.y_min,curr.z_min},{curr.x_max,curr.y_max,curr.z_max},
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


    }


};