// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file serialpatchtree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once


#include "core/io/logs.hpp"
#include "patch_field.hpp"
#include "patchtree.hpp"
#include "scheduler_mpi.hpp"
#include "core/sys/sycl_handler.hpp"
#include "aliases.hpp"
#include "patch_reduc_tree.hpp"
#include <tuple>

template<class fp_prec_vec>
struct PtNode{
    fp_prec_vec box_min;
    fp_prec_vec box_max;
    u64 childs_id0;
    u64 childs_id1;
    u64 childs_id2;
    u64 childs_id3;
    u64 childs_id4;
    u64 childs_id5;
    u64 childs_id6;
    u64 childs_id7;
};


template<class fp_prec_vec>
class SerialPatchTree{public:

    
    //TODO use unique pointer instead
    sycl::buffer<PtNode<fp_prec_vec>>* serial_tree_buf = nullptr;
    sycl::buffer<u64>*    linked_patch_ids_buf = nullptr;

    inline void attach_buf(){
        if(serial_tree_buf != nullptr) throw shamrock_exc("serial_tree_buf is already allocated");
        if(linked_patch_ids_buf != nullptr) throw shamrock_exc("linked_patch_ids_buf is already allocated");

        serial_tree_buf = new sycl::buffer<PtNode<fp_prec_vec>>(serial_tree.data(),serial_tree.size());
        linked_patch_ids_buf = new sycl::buffer<u64>(linked_patch_ids.data(),linked_patch_ids.size());
    }

    inline void detach_buf(){
        if(serial_tree_buf == nullptr) throw shamrock_exc("serial_tree_buf wasn't allocated");
        if(linked_patch_ids_buf == nullptr) throw shamrock_exc("linked_patch_ids_buf wasn't allocated");

        delete serial_tree_buf;
        serial_tree_buf = nullptr;
        
        delete linked_patch_ids_buf;
        linked_patch_ids_buf = nullptr;
    }



    
    



    private : 
    
    u32 level_count = 0;

    std::vector<PtNode<fp_prec_vec>> serial_tree;
    std::vector<u64> linked_patch_ids;

    void build_from_patch_tree(PatchTree &ptree, fp_prec_vec translate_factor, fp_prec_vec scale_factor);


    public: 

    inline SerialPatchTree(PatchTree &ptree, std::tuple<fp_prec_vec,fp_prec_vec> box_tranform){
        auto t = timings::start_timer("build serial ptree", timings::function);
        build_from_patch_tree(ptree, std::get<0>(box_tranform), std::get<1>(box_tranform));
        t.stop();
    }

    /**
     * @brief accesor to the number of level in the tree
     * 
     * @return const u32& number of level
     */
    inline const u32 & get_level_count(){
        return level_count;
    }

    /**
     * @brief accesor to the number of element in the tree
     * 
     * @return const u32& number of element
     */
    inline u32 get_element_count(){
        return serial_tree.size();
    }



    template<class type, class reduc_func>
    inline PatchFieldReduction<type> reduce_field(sycl::queue & queue,PatchScheduler & sched, PatchField<type> & pfield){

        PatchFieldReduction<type> predfield;

        std::cout << "resize to " << get_element_count() << std::endl;
        predfield.tree_field.resize(get_element_count());
        

        {
            auto lpid = linked_patch_ids_buf->get_access<sycl::access::mode::read>();

            //init reduction
            std::unordered_map<u64,u64> & idp_to_gid = sched.patch_list.id_patch_to_global_idx;
            for (u64 idx = 0; idx < get_element_count() ; idx ++) {
                predfield.tree_field[idx] = 
                    (lpid[idx] != u64_max) ? 
                    pfield.global_values[
                        idp_to_gid[lpid[idx]]
                    ]
                        : 
                    type()
                        ;

                //std::cout << " el " << idx << " " << predfield.tree_field[idx]  << std::endl;
            }
        }


        
        //std::cout << "predfield.attach_buf();" << std::endl;

        predfield.attach_buf();

        sycl::range<1> range{get_element_count()};

        u32 end_loop = get_level_count();

        for (u32 level = 0; level < end_loop; level ++) {

            // {
            //     auto f = predfield.tree_field_buf->template get_access<sycl::access::mode::read>();
            //     std::cout << "[";
            //     for (u64 idx = 0; idx < get_element_count() ; idx ++) {
            //         std::cout  << f[idx] << ",";
            //     }
            //     std::cout << std::endl;
            // }

            std::cout << "queue submit : " << level << " " <<end_loop << " " << (level < end_loop )<<std::endl;
            queue.submit([&](sycl::handler &cgh) {

                
                auto tree = this->serial_tree_buf->template get_access<sycl::access::mode::read>(cgh);
                
                auto f = predfield.tree_field_buf->template get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class OctreeReduction>(range, [=](sycl::item<1> item) {
                    u64 i = (u64)item.get_id(0);

                    u64 idx_c0 = tree[i].childs_id0;
                    u64 idx_c1 = tree[i].childs_id1;
                    u64 idx_c2 = tree[i].childs_id2;
                    u64 idx_c3 = tree[i].childs_id3;
                    u64 idx_c4 = tree[i].childs_id4;
                    u64 idx_c5 = tree[i].childs_id5;
                    u64 idx_c6 = tree[i].childs_id6;
                    u64 idx_c7 = tree[i].childs_id7;

                    if(idx_c0 != u64_max){
                        f[i] = reduc_func::reduce(
                                f[idx_c0],
                                f[idx_c1],
                                f[idx_c2],
                                f[idx_c3],
                                f[idx_c4],
                                f[idx_c5],
                                f[idx_c6],
                                f[idx_c7]
                            );
                    }
                    
                });
            });

            
        }
        // {
        //     auto f = predfield.tree_field_buf->template get_access<sycl::access::mode::read>();
        //     std::cout << "[";
        //     for (u64 idx = 0; idx < get_element_count() ; idx ++) {
        //         std::cout  << f[idx] << ",";
        //     }
        //     std::cout << std::endl;
        // }

        return predfield;

    }


    inline void dump_dat(){
        for (u64 idx = 0; idx < get_element_count() ; idx ++) {
            std::cout << idx << " (" << 
            serial_tree[idx].childs_id0  << ", "<< 
            serial_tree[idx].childs_id1  << ", "<< 
            serial_tree[idx].childs_id2  << ", "<< 
            serial_tree[idx].childs_id3  << ", "<< 
            serial_tree[idx].childs_id4  << ", "<< 
            serial_tree[idx].childs_id5  << ", "<< 
            serial_tree[idx].childs_id6  << ", "<< 
            serial_tree[idx].childs_id7  << ", "
            << ")" << std::endl;
        }
    }



};
