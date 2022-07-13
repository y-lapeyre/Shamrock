// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "aliases.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "core/utils/geometry_utils.hpp"

template <class posvec, class kername>
inline sycl::buffer<u64> __compute_object_patch_owner(sycl::queue &queue, sycl::buffer<posvec> &position_buffer,
                                                      SerialPatchTree<posvec> &sptree) {

    sycl::buffer<u64> new_owned_id(position_buffer.size());

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

        cgh.parallel_for<kername>(sycl::range(position_buffer.size()), [=](sycl::item<1> item) {
            u32 i = (u32)item.get_id(0);

            // TODO implement the version with multiple roots
            u64 current_node = 0;
            u64 result_node  = u64_max;

            auto xyz = pos[i];

            //out << i << " : \n";

            for (u32 step = 0; step < max_lev+1; step++) {
                PtNode<posvec> cur_node = tnode[current_node];

                if (cur_node.childs_id0 != u64_max) {

                    if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id0].box_min,
                                                   tnode[cur_node.childs_id0].box_max)) {
                        current_node = cur_node.childs_id0;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id1].box_min,
                                                          tnode[cur_node.childs_id1].box_max)) {
                        current_node = cur_node.childs_id1;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id2].box_min,
                                                          tnode[cur_node.childs_id2].box_max)) {
                        current_node = cur_node.childs_id2;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id3].box_min,
                                                          tnode[cur_node.childs_id3].box_max)) {
                        current_node = cur_node.childs_id3;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id4].box_min,
                                                          tnode[cur_node.childs_id4].box_max)) {
                        current_node = cur_node.childs_id4;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id5].box_min,
                                                          tnode[cur_node.childs_id5].box_max)) {
                        current_node = cur_node.childs_id5;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id6].box_min,
                                                          tnode[cur_node.childs_id6].box_max)) {
                        current_node = cur_node.childs_id6;
                    } else if (BBAA::is_particle_in_patch(xyz, tnode[cur_node.childs_id7].box_min,
                                                          tnode[cur_node.childs_id7].box_max)) {
                        current_node = cur_node.childs_id7;
                    }
                    //out << current_node << " " << linked_node_id[current_node] << "\n";
                } else {
                    // out << "-> result\n";
                    result_node = linked_node_id[current_node];
                    break;
                }

                
            }

            //out << "-> " << current_node << " " << linked_node_id[current_node] << "\n";

            new_id[i] = result_node;
        });
    });

    return new_owned_id;
}