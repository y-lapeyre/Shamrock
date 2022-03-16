/**
 * @file patch_reduc_tree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once




#include <stdexcept>
#include <vector>



#include "sys/sycl_handler.hpp"

#include "patch/patchtree.hpp"

template<class type>
class PatchFieldReduction{public:

    std::vector<type> tree_field;
    sycl::buffer<type>* tree_field_buf = nullptr;


    inline void attach_buf(){
        if(tree_field_buf != nullptr) throw std::runtime_error("tree_field_buf is already allocated");
        tree_field_buf = new sycl::buffer<type>(tree_field);
    }

    inline void detach_buf(){
        if(tree_field_buf == nullptr) throw std::runtime_error("tree_field_buf wasn't allocated");
        delete tree_field_buf;
        tree_field_buf = nullptr;
    }

    
    // inline void octtree_reduction(
    //     sycl::queue & queue, 
    //     SerialPatchTree<box_vectype> & sptree,
    //     SchedulerMPI & sched){
        
    //     std::unordered_map<u64,u64> & idp_to_gid = sched.patch_list.id_patch_to_global_idx;

    //     cl::sycl::range<1> range{sptree.get_element_count()};

    //     for (u32 level = 0; level < sptree.get_level_count(); level ++) {
    //         queue.submit([&](cl::sycl::handler &cgh) {
                
                

    //             cgh.parallel_for<class OctreeReduction>(range, [=](cl::sycl::item<1> item) {
    //                 u64 i = (u64)item.get_id(0);


                    
    //             });
    //         });
    //     }

    // }

};