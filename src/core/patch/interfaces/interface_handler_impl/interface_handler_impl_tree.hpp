#pragma once

#include "core/tree/radix_tree.hpp"
#include "interface_handler_impl_list.hpp"


//%Impl status : Clean unfinished














template<class pos_prec, class u_morton> 
class Interfacehandler<Tree_Send,pos_prec,Radix_Tree<u_morton, sycl::vec<pos_prec, 3>>>{

    

    public:

    using flt = pos_prec;
    using vec = sycl::vec<flt, 3>;

    private : 

    using CutTree = typename Radix_Tree<u_morton, vec>::CuttedTree;

    //Store the result of a tree cut
    struct CommListing {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        CutTree cutted_tree;
    };

    public:

    // for now interact crit has shape (vec,vec) -> bool 
    // in order to pass for exemple h max we need a full tree field (patch field + radix tree field) 
    template<class Func_interactcrit>
    inline void compute_interface_list(
        PatchScheduler &sched, 
        SerialPatchTree<vec> sptree, 
        Func_interactcrit && interact_crit, 
        bool periodic){


        


    }

    //TODO
    void initial_fetch();

    void fetch_field();


    template<class Function> void for_each_interface(u64 patch_id, Function && fct);
};



