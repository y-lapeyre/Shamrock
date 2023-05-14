// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "aliases.hpp"
#include "shambase/string.hpp"
#include "shamrock/legacy/patch/utility/serialpatchtree.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamsys/legacy/log.hpp"

template <class posvec, class kername>
inline sycl::buffer<u64> __compute_object_patch_owner(sycl::queue &queue, sycl::buffer<posvec> &position_buffer, u32 len,
                                                      SerialPatchTree<posvec> &sptree) {

    sycl::buffer<u64> new_owned_id(len);

    using namespace shamrock::patch;

    // std::cout << "linked id state :\n";
    // {
    //     auto lpid = sptree.linked_patch_ids_buf->template get_access<sycl::access::mode::read>();
    //     for (u32 i = 0 ; i < sptree.linked_patch_ids_buf->size(); i++) {
    //         std::cout << lpid[i] << " ";
    //     }
    // }

    queue.submit([&](sycl::handler &cgh) {
        auto pos            = position_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto tnode          = sptree.serial_tree_buf->template get_access<sycl::access::mode::read>(cgh);
        auto linked_node_id = sptree.linked_patch_ids_buf->template get_access<sycl::access::mode::read>(cgh);
        auto new_id         = new_owned_id.get_access<sycl::access::mode::discard_write>(cgh);

        auto max_lev = sptree.get_level_count();

        //sycl::stream out(1024, 768, cgh);

        cgh.parallel_for<kername>(sycl::range(len), [=](sycl::item<1> item) {
            u32 i = (u32)item.get_id(0);

            // TODO implement the version with multiple roots
            u64 current_node = 0;
            u64 result_node  = u64_max;

            auto xyz = pos[i];

            //out << i << " : \n";

            using PtNode = shamrock::scheduler::SerialPatchNode<posvec>;

            for (u32 step = 0; step < max_lev+1; step++) {
                PtNode cur_node = tnode[current_node];

                if (cur_node.childs_id[0] != u64_max) {

                    if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[0]].box_min,
                                                   tnode[cur_node.childs_id[0]].box_max)) {
                        current_node = cur_node.childs_id[0];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[1]].box_min,
                                                          tnode[cur_node.childs_id[1]].box_max)) {
                        current_node = cur_node.childs_id[1];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[2]].box_min,
                                                          tnode[cur_node.childs_id[2]].box_max)) {
                        current_node = cur_node.childs_id[2];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[3]].box_min,
                                                          tnode[cur_node.childs_id[3]].box_max)) {
                        current_node = cur_node.childs_id[3];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[4]].box_min,
                                                          tnode[cur_node.childs_id[4]].box_max)) {
                        current_node = cur_node.childs_id[4];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[5]].box_min,
                                                          tnode[cur_node.childs_id[5]].box_max)) {
                        current_node = cur_node.childs_id[5];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[6]].box_min,
                                                          tnode[cur_node.childs_id[6]].box_max)) {
                        current_node = cur_node.childs_id[6];
                    } else if (Patch::is_in_patch_converted(xyz, tnode[cur_node.childs_id[7]].box_min,
                                                          tnode[cur_node.childs_id[7]].box_max)) {
                        current_node = cur_node.childs_id[7];
                    }
                    //out << current_node << " " << linked_node_id[current_node] << "\n";
                } else {
                    // out << "-> result\n";
                    
                    
                    result_node = linked_node_id[current_node];
                    break;
                }

                
            }
            
            if constexpr(false){
                PtNode cur_node = tnode[current_node];
                if(xyz.z()==0){
                    logger::raw(
                        shambase::format("{:5} ({}) -> {} [{} {}]\n", 
                            i,
                            Patch::is_in_patch_converted(xyz, cur_node.box_min , cur_node.box_max),
                            xyz.z(),
                            cur_node.box_min.z() , 
                            cur_node.box_max.z()
                        )
                    );
                }
            }

            new_id[i] = result_node;
        });
    });

    return new_owned_id;
}