// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "radix_tree.hpp"


#include "access/access.hpp"
#include "aliases.hpp"
#include "buffer.hpp"
#include "kernels/morton_kernels.hpp"
#include "properties/accessor_properties.hpp"
#include "range.hpp"
#include <tuple>
#include <vector>





template <class u_morton, class vec3>
Radix_Tree<u_morton, vec3>::Radix_Tree(
    sycl::queue &queue, std::tuple<vec3, vec3> treebox, std::unique_ptr<sycl::buffer<vec3>> &pos_buf, u32 cnt_obj
) {
    if (cnt_obj > i32_max - 1) {
        throw shamrock_exc("number of element in patch above i32_max-1");
    }

    logger::debug_sycl_ln("RadixTree", "box dim :", std::get<0>(treebox), std::get<1>(treebox));

    box_coord = treebox;

    u32 morton_len = get_next_pow2_val(cnt_obj);
    logger::debug_sycl_ln("RadixTree", "morton buffer lenght :", morton_len);

    buf_morton = std::make_unique<sycl::buffer<u_morton>>(morton_len);

    logger::debug_sycl_ln("RadixTree", "xyz to morton");
    sycl_xyz_to_morton<u_morton, vec3>(queue, cnt_obj, pos_buf, std::get<0>(box_coord), std::get<1>(box_coord), buf_morton);

    logger::debug_sycl_ln("RadixTree", "fill trailling buffer");
    sycl_fill_trailling_buffer<u_morton>(queue, cnt_obj, morton_len, buf_morton);

    logger::debug_sycl_ln("RadixTree", "sorting morton buffer");
    buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(morton_len);

    queue.submit([&](sycl::handler &cgh) {
        auto pidm = buf_particle_index_map->get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for(sycl::range(morton_len), [=](sycl::item<1> item) { pidm[item] = item.get_id(0); });
    });

    sycl_sort_morton_key_pair(queue, morton_len, buf_particle_index_map, buf_morton);

    // return a sycl buffer from reduc index map instead
    logger::debug_sycl_ln("RadixTree", "reduction algorithm"); // TODO put reduction level in class member
    std::vector<u32> reduc_index_map;
    reduction_alg(queue, cnt_obj, buf_morton, 5, reduc_index_map, tree_leaf_count);

    logger::debug_sycl_ln(
        "RadixTree", "reduction results : (before :", cnt_obj, " | after :", tree_leaf_count,
        ") ratio :", format("%2.2f", f32(cnt_obj) / f32(tree_leaf_count))
    );

    if (tree_leaf_count > 1) {

        //buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map.data(),reduc_index_map.size());

        {
            sycl::buffer<u32> tmp (reduc_index_map.data(), reduc_index_map.size());
            buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map.size());
        
            queue.submit([&](sycl::handler &cgh) {
                auto source = tmp.get_access<sycl::access::mode::read>(cgh);
                auto dest = buf_reduc_index_map->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for(sycl::range(reduc_index_map.size()), [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        
        }
        

        

        logger::debug_sycl_ln("RadixTree", "sycl_morton_remap_reduction");
        buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(tree_leaf_count);

        sycl_morton_remap_reduction(queue, tree_leaf_count, buf_reduc_index_map, buf_morton, buf_tree_morton);

        tree_internal_count = tree_leaf_count - 1;

        buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
        buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
        buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
        buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
        buf_endrange    = std::make_unique<sycl::buffer<u32>>(tree_internal_count);

        sycl_karras_alg(
            queue, tree_internal_count, buf_tree_morton, buf_lchild_id, buf_rchild_id, buf_lchild_flag, buf_rchild_flag,
            buf_endrange
        );

        one_cell_mode = false;
    } else if (tree_leaf_count == 1) {
        // throw shamrock_exc("one cell mode is not implemented");
        // TODO do some extensive test on one cell mode
        one_cell_mode = true;

        tree_internal_count = 1;
        tree_leaf_count     = 2;
        reduc_index_map.push_back(0);

        buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map.data(), reduc_index_map.size());

        buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
        buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
        buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
        buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
        buf_endrange    = std::make_unique<sycl::buffer<u32>>(tree_internal_count);

        {
            auto rchild_id   = (buf_rchild_id->get_access<sycl::access::mode::discard_write>());
            auto lchild_id   = (buf_lchild_id->get_access<sycl::access::mode::discard_write>());
            auto rchild_flag = (buf_rchild_flag->get_access<sycl::access::mode::discard_write>());
            auto lchild_flag = (buf_lchild_flag->get_access<sycl::access::mode::discard_write>());
            auto endrange    = (buf_endrange->get_access<sycl::access::mode::discard_write>());

            rchild_id[0]   = 0;
            lchild_id[0]   = 1;
            rchild_flag[0] = 1;
            lchild_flag[0] = 1;
            endrange[0]    = 1;
        }

    } else {
        throw shamrock_exc("empty patch should be skipped");
    }
}





template <class u_morton, class vec3> void Radix_Tree<u_morton, vec3>::compute_cellvolume(sycl::queue &queue) {
    if (!one_cell_mode) {

        logger::debug_sycl_ln("RadixTree", "compute_cellvolume");

        buf_pos_min_cell = std::make_unique<sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);
        buf_pos_max_cell = std::make_unique<sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);

        sycl_compute_cell_ranges(
            queue, tree_leaf_count, tree_internal_count, buf_tree_morton, buf_lchild_id, buf_rchild_id, buf_lchild_flag,
            buf_rchild_flag, buf_endrange, buf_pos_min_cell, buf_pos_max_cell
        );

    } else {
        // throw shamrock_exc("one cell mode is not implemented");
        // TODO do some extensive test on one cell mode

        buf_pos_min_cell = std::make_unique<sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);
        buf_pos_max_cell = std::make_unique<sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);

        {
            auto pos_min_cell = buf_pos_min_cell->template get_access<sycl::access::mode::discard_write>();
            auto pos_max_cell = buf_pos_max_cell->template get_access<sycl::access::mode::discard_write>();

            pos_min_cell[0] = {0};
            pos_max_cell[0] = {morton_3d::morton_types<u_morton>::max_val};

            pos_min_cell[1] = {0};
            pos_max_cell[1] = {morton_3d::morton_types<u_morton>::max_val};

            pos_min_cell[2] = {0};
            pos_max_cell[2] = {0};
        }
    }

    buf_pos_min_cell_flt = std::make_unique<sycl::buffer<vec3>>(tree_internal_count + tree_leaf_count);
    buf_pos_max_cell_flt = std::make_unique<sycl::buffer<vec3>>(tree_internal_count + tree_leaf_count);

    logger::debug_sycl_ln("RadixTree", "sycl_convert_cell_range");

    sycl_convert_cell_range<u_morton, vec3>(
        queue, tree_leaf_count, tree_internal_count, std::get<0>(box_coord), std::get<1>(box_coord), buf_pos_min_cell,
        buf_pos_max_cell, buf_pos_min_cell_flt, buf_pos_max_cell_flt
    );
}






template <class u_morton, class vec3>
void Radix_Tree<u_morton, vec3>::compute_int_boxes(
    sycl::queue &queue, std::unique_ptr<sycl::buffer<flt>> &int_rad_buf, flt tolerance
) {

    logger::debug_sycl_ln("RadixTree", "compute int boxes");

    buf_cell_interact_rad = std::make_unique<sycl::buffer<flt>>(tree_internal_count + tree_leaf_count);
    sycl::range<1> range_leaf_cell{tree_leaf_count};

    queue.submit([&](sycl::handler &cgh) {
        u32 offset_leaf = tree_internal_count;

        auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::discard_write>(cgh);
        auto h          = int_rad_buf->template get_access<sycl::access::mode::read>(cgh);

        auto cell_particle_ids  = buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
        auto particle_index_map = buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);

        flt tol = tolerance;

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            u32 min_ids = cell_particle_ids[gid];
            u32 max_ids = cell_particle_ids[gid + 1];
            f32 h_tmp   = 0;

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                f32 h_a = h[particle_index_map[id_s]] * tol;
                h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
            }

            h_max_cell[offset_leaf + gid] = h_tmp;
        });
    });

    sycl::range<1> range_tree{tree_internal_count};
    auto ker_reduc_hmax = [&](sycl::handler &cgh) {
        u32 offset_leaf = tree_internal_count;

        auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::read_write>(cgh);

        auto rchild_id   = buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
        auto lchild_id   = buf_lchild_id->get_access<sycl::access::mode::read>(cgh);
        auto rchild_flag = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto lchild_flag = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
            u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

            flt h_l = h_max_cell[lid];
            flt h_r = h_max_cell[rid];

            h_max_cell[gid] = (h_r > h_l ? h_r : h_l);
        });
    };

    for (u32 i = 0; i < tree_depth; i++) {
        queue.submit(ker_reduc_hmax);
    }
}

template <class u_morton, class vec3>
std::tuple<Radix_Tree<u_morton, vec3>, PatchData> Radix_Tree<u_morton, vec3>::cut_tree(
    sycl::queue &queue, const std::tuple<vec3, vec3> &cut_range, const PatchData &pdat_source
) {

    Radix_Tree<u_morton, vec3> ret_tree;
    ret_tree.box_coord = box_coord;


    u32 total_count             = tree_internal_count + tree_leaf_count;
    sycl::range<1> range_tree{total_count};

    auto init_valid_buf = [&]() -> sycl::buffer<u8>{
        
        sycl::buffer<u8> valid_node = sycl::buffer<u8>(total_count);

        sycl::range<1> range_tree{total_count};

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_valid_node{valid_node, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor acc_pos_cell_min{*buf_pos_min_cell_flt, cgh, sycl::read_only};
            sycl::accessor acc_pos_cell_max{*buf_pos_max_cell_flt, cgh, sycl::read_only};

            vec3 v_max = std::get<0>(cut_range);
            vec3 v_min = std::get<1>(cut_range);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                acc_valid_node[item] = BBAA::cella_neigh_b(v_min, v_max, acc_pos_cell_min[item], acc_pos_cell_max[item]);
            });
        });

        return valid_node;
    };


    

    {

        
        sycl::buffer<u8> valid_node = init_valid_buf();

        //flag 1 valid
        //flag 0 to be deleted 
        //flag 2 anything below should be deleted (2 if initialy 0 & parent = 1)
        // basically 2 is le thing that would end up in the excluded lambda part

        {// cascade zeros down the tree

            sycl::buffer<u8> valid_node_new = sycl::buffer<u8>(total_count);

            for(u32 it = 0; it < tree_depth; it++){
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_valid_node_old{valid_node, cgh, sycl::read_only};
                    sycl::accessor acc_valid_node_new{valid_node_new, cgh, sycl::write_only,sycl::no_init};

                    sycl::accessor acc_pos_cell_min{*buf_pos_min_cell_flt, cgh, sycl::read_only};
                    sycl::accessor acc_pos_cell_max{*buf_pos_max_cell_flt, cgh, sycl::read_only};

                    sycl::accessor acc_lchild_id   {*buf_lchild_id  ,cgh,sycl::read_only};
                    sycl::accessor acc_rchild_id   {*buf_rchild_id  ,cgh,sycl::read_only};
                    sycl::accessor acc_lchild_flag {*buf_lchild_flag,cgh,sycl::read_only};
                    sycl::accessor acc_rchild_flag {*buf_rchild_flag,cgh,sycl::read_only};

                    u32 leaf_offset = tree_internal_count;

                    cgh.parallel_for(sycl::range<1>(tree_internal_count), [=](sycl::item<1> item) {

                        u32 lid = acc_lchild_id[item] + leaf_offset * acc_lchild_flag[item];
                        u32 rid = acc_rchild_id[item] + leaf_offset * acc_rchild_flag[item];

                        u8 old_nid_falg = acc_valid_node_old[item];

                        if(item.get_linear_id() == 0){
                            acc_valid_node_new[item] = old_nid_falg;
                        }

                        if(old_nid_falg == 0 || old_nid_falg == 2){
                            acc_valid_node_new[lid] = 0;
                            acc_valid_node_new[rid] = 0;
                        }else{
                            u8 old_lid_falg = acc_valid_node_old[lid];
                            u8 old_rid_falg = acc_valid_node_old[rid];

                            if(old_lid_falg == 0) { old_lid_falg = 2; }
                            if(old_rid_falg == 0) { old_rid_falg = 2; }

                            acc_valid_node_new[lid] = old_lid_falg;
                            acc_valid_node_new[rid] = old_rid_falg;
                        }

                    });
                });

                std::swap(valid_node,valid_node_new);
            }

        }



        sycl::buffer<u32> node_id_remap(total_count);

        ret_tree.tree_internal_count = 0;
        ret_tree.tree_leaf_count = 0;
        

        {//count leafs and nodes
            sycl::host_accessor acc_valid_node{valid_node, sycl::read_only};
            sycl::host_accessor acc_node_id_remap{valid_node, sycl::write_only, sycl::no_init};

            u32 next_id = 0;

            for(u32 i = 0; i < tree_internal_count; i++){
                if(acc_valid_node[i]){
                    acc_node_id_remap[i] = next_id;
                    ret_tree.tree_internal_count ++;
                    next_id ++;
                }else{
                    acc_node_id_remap[i] = u32_max;
                }
            }

            for(u32 i = tree_internal_count; i < total_count; i++){
                if(acc_valid_node[i]){
                    acc_node_id_remap[i] = next_id;
                    ret_tree.tree_leaf_count ++;
                    next_id ++;
                }else{
                    acc_node_id_remap[i] = u32_max;
                }
            }
        }




        ret_tree.buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(ret_tree.tree_leaf_count);

        ret_tree.buf_lchild_id = std::make_unique<sycl::buffer<u32>>(ret_tree.tree_internal_count);  
        ret_tree.buf_rchild_id = std::make_unique<sycl::buffer<u32>>(ret_tree.tree_internal_count);  
        ret_tree.buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(ret_tree.tree_internal_count);
        ret_tree.buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(ret_tree.tree_internal_count);
        ret_tree.buf_endrange = std::make_unique<sycl::buffer<u32>>(ret_tree.tree_internal_count);   

        ret_tree.buf_pos_min_cell     = std::make_unique<sycl::buffer<vec3i>>(ret_tree.tree_leaf_count+ret_tree.tree_internal_count);     
        ret_tree.buf_pos_max_cell     = std::make_unique<sycl::buffer<vec3i>>(ret_tree.tree_leaf_count+ret_tree.tree_internal_count);     
        ret_tree.buf_pos_min_cell_flt = std::make_unique<sycl::buffer<vec3>> (ret_tree.tree_leaf_count+ret_tree.tree_internal_count); 
        ret_tree.buf_pos_max_cell_flt = std::make_unique<sycl::buffer<vec3>> (ret_tree.tree_leaf_count+ret_tree.tree_internal_count); 


        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor remap_nid{node_id_remap, cgh, sycl::read_only};

            sycl::accessor new_buf_tree_morton {*ret_tree.buf_tree_morton, cgh, sycl::write_only,sycl::no_init};
            sycl::accessor old_buf_tree_morton {*buf_tree_morton, cgh, sycl::read_only,sycl::no_init};

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                
                u32 new_nid = remap_nid[item];

                if (new_nid != u32_max){
                    new_buf_tree_morton[new_nid] = old_buf_tree_morton[item];
                }

            });
        });

        // TODO code extraction part
    }
}















template class Radix_Tree<u32, f32_3>;
template class Radix_Tree<u64, f32_3>;

template class Radix_Tree<u32, f64_3>;
template class Radix_Tree<u64, f64_3>;