// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "radix_tree.hpp"


#include "aliases.hpp"
#include "kernels/morton_kernels.hpp"
#include <tuple>
#include <vector>





template <class u_morton, class vec3>
Radix_Tree<u_morton, vec3>::Radix_Tree(
    sycl::queue &queue, std::tuple<vec3, vec3> treebox, std::unique_ptr<sycl::buffer<vec3>> &pos_buf, u32 cnt_obj, u32 reduc_level
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
    reduction_alg(queue, cnt_obj, buf_morton, reduc_level, reduc_index_map, tree_leaf_count);

    u32 reduc_index_map_len = reduc_index_map.size();

    logger::debug_sycl_ln(
        "RadixTree", "reduction results : (before :", cnt_obj, " | after :", tree_leaf_count,
        ") ratio :", format("%2.2f", f32(cnt_obj) / f32(tree_leaf_count))
    );

    if (tree_leaf_count > 1) {

        //buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map.data(),reduc_index_map.size());

        {

            sycl::buffer<u32> tmp (reduc_index_map.data(), reduc_index_map_len);
                
            buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map_len);
        
            queue.submit([&](sycl::handler &cgh) {
                auto source = tmp.get_access<sycl::access::mode::read>(cgh);
                auto dest = buf_reduc_index_map->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for(sycl::range(reduc_index_map_len), [=](sycl::item<1> item) { dest[item] = source[item]; });
            });

            //HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer 
            //doesn't wait for the end of the queue resulting in out of bound access
            #ifdef SYCL_COMP_HIPSYCL
            queue.wait();
            #endif

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



template<class T>
std::string print_member(const T & a);

template<> std::string print_member(const u8 & a){
    return format("%d",u32(a));
}

template<> std::string print_member(const u32 & a){
    return format("%d",a);
}


template <class u_morton, class vec3> template<class T>
void Radix_Tree<u_morton, vec3>::print_tree_field(sycl::buffer<T> & buf_field){

    sycl::host_accessor acc{buf_field, sycl::read_only};

    u32 total_count             = tree_internal_count + tree_leaf_count;

    u32 offset_leaf = tree_internal_count;

    sycl::host_accessor rchild_id   {*buf_rchild_id  };
    sycl::host_accessor lchild_id   {*buf_lchild_id  };
    sycl::host_accessor rchild_flag {*buf_rchild_flag};
    sycl::host_accessor lchild_flag {*buf_lchild_flag};


    auto printer = [&](){

        auto get_print_step = [&](u32 gid, std::string prefix, bool is_left, auto & step_ref) -> std::string{

            std::string ret_val = "";

            if(!is_left){
                ret_val += prefix;
            }

            std::string val = " (" + print_member(acc[gid])+ ") ";
            std::string val_empt = std::string(val.size(),' ');

            ret_val += (is_left ? "╦══" : "╚══" );
            ret_val += val;

            if (gid < offset_leaf){
                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                ret_val += step_ref(lid, prefix + (is_left ? "║  " + val_empt: "   " + val_empt), true ,step_ref);
                ret_val += step_ref(rid, prefix + (is_left ? "║  " + val_empt: "   " + val_empt), false,step_ref);
            }else{
                ret_val += "\n";
            }


            return ret_val;
            

        };


        logger::raw_ln(get_print_step(0,"",false,get_print_step));
    };

    printer();
    

}


template void Radix_Tree<u32, f64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void Radix_Tree<u32, f32_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void Radix_Tree<u64, f64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void Radix_Tree<u64, f32_3>::print_tree_field(sycl::buffer<u32> &buf_field);













template <class u_morton, class vec3>
std::tuple<Radix_Tree<u_morton, vec3>,std::unique_ptr<sycl::buffer<u32>>, PatchData> Radix_Tree<u_morton, vec3>::cut_tree(
    sycl::queue &queue, const std::tuple<vec3, vec3> &cut_range, const PatchData &pdat_source
) {

    


    u32 total_count             = tree_internal_count + tree_leaf_count;
    sycl::range<1> range_tree{total_count};

    auto init_valid_buf = [&]() -> sycl::buffer<u8>{
        
        sycl::buffer<u8> valid_node = sycl::buffer<u8>(total_count);

        sycl::range<1> range_tree{total_count};

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_valid_node{valid_node, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor acc_pos_cell_min{*buf_pos_min_cell_flt, cgh, sycl::read_only};
            sycl::accessor acc_pos_cell_max{*buf_pos_max_cell_flt, cgh, sycl::read_only};

            vec3 v_min = std::get<0>(cut_range);
            vec3 v_max = std::get<1>(cut_range);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                acc_valid_node[item] = BBAA::cella_neigh_b(v_min, v_max, acc_pos_cell_min[item], acc_pos_cell_max[item]);
            });
        });

        return valid_node;
    };


    


    

    {        

        logger::debug_sycl_ln("Radixtree", "computing valid node buf");

        sycl::buffer<u8> valid_node = init_valid_buf();

        

        //flag 1 valid
        //flag 0 to be deleted 
        //flag 2 anything below should be deleted (2 if initialy 0 & parent = 1)
        // basically 2 is le thing that would end up in the excluded lambda part

        {// cascade zeros down the tree

            sycl::buffer<u8> valid_node_new = sycl::buffer<u8>(total_count);

            for(u32 it = 0; it < tree_depth; it++){


                logger::debug_sycl_ln("Radixtree", "cascading zeros step : ",it);
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_valid_node_old{valid_node, cgh, sycl::read_only};
                    sycl::accessor acc_valid_node_new{valid_node_new, cgh, sycl::write_only,sycl::no_init};

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


        {
            logger::debug_sycl_ln("Radixtree", "valid_node_state");
            print_tree_field(valid_node);
            logger::raw_ln("");
        }


        sycl::buffer<u8> valid_tree_morton (tree_leaf_count);

        auto print_valid_morton = [&]{
            logger::debug_sycl_ln("Radixtree", "valid_tree_morton");

            sycl::buffer<u32> print_map(total_count);

            {

            
                sycl::host_accessor acc {print_map};
                sycl::host_accessor acc_leaf {valid_tree_morton};

                for(u32 i = 0; i < tree_leaf_count; i++){
                    acc[i + tree_internal_count] = acc_leaf[i];
                }
                
                for(u32 i = 0; i < tree_internal_count; i++){
                    acc[i] = acc_leaf[i];
                }

            }
            
            print_tree_field(print_map);



            logger::raw_ln("");
        };

        queue.submit([&](sycl::handler &cgh) {

            sycl::accessor acc_valid_tree_morton {valid_tree_morton, cgh,sycl::write_only,sycl::no_init};

            sycl::accessor acc_valid_node{valid_node, cgh, sycl::read_only};

            u32 leaf_offset = tree_internal_count;

            cgh.parallel_for(sycl::range<1>(tree_leaf_count), [=](sycl::item<1> item) {

                u8 leaf_val = acc_valid_node[item.get_linear_id() + leaf_offset];

                if(item.get_linear_id() < leaf_offset){
                    if(acc_valid_node[item] == 2){
                        leaf_val = 2;
                    }
                }

                acc_valid_tree_morton[item] = leaf_val;

            });
        });

        print_valid_morton();









        //generate the new tree

        Radix_Tree ret;

        ret.box_coord = box_coord;





        std::vector<u32> extract_id;

        {

            std::vector<u_morton> new_buf_morton;
            std::vector<u32> new_buf_particle_index_map;
            std::vector<u32> new_reduc_index_map;

            u32 leaf_offset = tree_internal_count;

            sycl::host_accessor cell_index_map{*buf_reduc_index_map,sycl::read_only};
            sycl::host_accessor particle_index_map{*buf_particle_index_map,sycl::read_only};

            sycl::host_accessor acc_valid_tree_morton {valid_tree_morton,sycl::read_only};

            sycl::host_accessor acc_morton {*buf_morton, sycl::read_only};

            u32 cnt = 0;

            for(u32 i = 0; i < tree_leaf_count; i++){
                if(acc_valid_tree_morton[i] != 0){

                    {
                        // loop on particle indexes
                        uint min_ids = cell_index_map[i     ];
                        uint max_ids = cell_index_map[i + 1 ];

                        new_reduc_index_map.push_back(cnt);

                        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                            //recover old index before morton sort
                            uint id_b = particle_index_map[id_s];

                            //iteration function
                            {
                                extract_id.push_back(id_b);
                                new_buf_morton.push_back(acc_morton[id_b]);
                                new_buf_particle_index_map.push_back(cnt);

                                cnt++;
                            }
                        }


                    }
                }
                
            }

            new_reduc_index_map.push_back(cnt);






            std::vector<u_morton> new_morton_tree;

            {
                sycl::host_accessor acc_tree_morton {*buf_tree_morton};

                sycl::host_accessor acc_valid_tree_morton {valid_tree_morton,sycl::read_only};

                for(u32 i = 0; i < tree_leaf_count; i++){
                    if(acc_valid_tree_morton[i] != 0){
                        new_morton_tree.push_back(acc_tree_morton[i]);
                    }
                    
                }
            }








            ret.tree_leaf_count = new_morton_tree.size();
            ret.tree_internal_count = ret.tree_leaf_count -1;

            ret.buf_morton = std::make_unique<sycl::buffer<u_morton>>(new_buf_morton.size());
            {
                sycl::host_accessor acc{* ret.buf_morton, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_buf_morton.size(); i++) {
                    acc[i] = new_buf_morton[i];
                }
            }


            ret.buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(new_buf_particle_index_map.size());
            {
                sycl::host_accessor acc{* ret.buf_particle_index_map, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_buf_particle_index_map.size(); i++) {
                    acc[i] = new_buf_particle_index_map[i];
                }
            }

            if(ret.tree_leaf_count > 1){

                    
                ret.buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(new_reduc_index_map.size());
                {
                    sycl::host_accessor acc{* ret.buf_reduc_index_map, sycl::write_only, sycl::no_init};
                    for (u32 i = 0 ; i < new_reduc_index_map.size(); i++) {
                        acc[i] = new_reduc_index_map[i];
                    }
                }


                ret.buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(new_morton_tree.size());
                {
                    sycl::host_accessor acc{* ret.buf_tree_morton, sycl::write_only, sycl::no_init};
                    for (u32 i = 0 ; i < new_morton_tree.size(); i++) {
                        acc[i] = new_morton_tree[i];
                    }
                }

            
                ret.buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(ret.tree_internal_count);
                ret.buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(ret.tree_internal_count);
                ret.buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(ret.tree_internal_count);
                ret.buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(ret.tree_internal_count);
                ret.buf_endrange    = std::make_unique<sycl::buffer<u32>>(ret.tree_internal_count);

                sycl_karras_alg(
                    queue, ret.tree_internal_count, ret.buf_tree_morton, ret.buf_lchild_id, ret.buf_rchild_id, ret.buf_lchild_flag, ret.buf_rchild_flag,
                    ret.buf_endrange
                );

                one_cell_mode = false;
            }else{
                throw ShamrockSyclException("not implemented");
            }
        }


        ret.compute_cellvolume(queue);



        logger::raw_ln("len new tree",ret.tree_internal_count);

        #if false
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_naive = std::make_unique<sycl::buffer<u32>>(ret.tree_leaf_count + ret.tree_internal_count);

        {
            auto & new_node_id_to_old = new_node_id_to_old_naive;

            //junk fill
            {
                sycl::host_accessor acc{* new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }

            
            sycl::host_accessor acc_new_node_id_to_old {*new_node_id_to_old,sycl::write_only, sycl::no_init};
            
            sycl::host_accessor new_tree_acc_pos_min_cell{*ret.buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor new_tree_acc_pos_max_cell{*ret.buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_acc_pos_min_cell{*buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor old_tree_acc_pos_max_cell{*buf_pos_max_cell,sycl::read_only};
            
            for(u32 i = 0 ; i < ret.tree_leaf_count + ret.tree_internal_count; i++){

                vec3i cur_pos_min_cell_a = new_tree_acc_pos_min_cell[i];
                vec3i cur_pos_max_cell_a = new_tree_acc_pos_max_cell[i];

                for(u32 j = 0 ; j < tree_leaf_count + tree_internal_count; j++){
                
                    vec3i cur_pos_min_cell_b = old_tree_acc_pos_min_cell[j];
                    vec3i cur_pos_max_cell_b = old_tree_acc_pos_max_cell[j];


                    auto is_same_box = [&]() -> bool{
                        return 
                            (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x()) && 
                            (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y()) && 
                            (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z()) && 
                            (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x()) && 
                            (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y()) && 
                            (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z()) ; 
                    };

                    if(is_same_box()){

                        u32 store_val = j;

                        logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr ->",cur_pos_min_cell_b,cur_pos_max_cell_b);


                        if(store_val >= tree_internal_count){
                            store_val -= tree_internal_count;
                        }

                        acc_new_node_id_to_old[i] = store_val;

                        break;
                    }


                }
            }
        }

        ret.print_tree_field(*new_node_id_to_old_naive);
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_v1 = std::make_unique<sycl::buffer<u32>>(ret.tree_leaf_count + ret.tree_internal_count);

        {
            auto & new_node_id_to_old = new_node_id_to_old_v1;

            //junk fill
            {
                sycl::host_accessor acc{* new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }

            
            sycl::host_accessor acc_new_node_id_to_old {*new_node_id_to_old,sycl::write_only, sycl::no_init};
            
            sycl::host_accessor new_tree_acc_pos_min_cell{*ret.buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor new_tree_acc_pos_max_cell{*ret.buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_acc_pos_min_cell{*buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor old_tree_acc_pos_max_cell{*buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_lchild_id   {*buf_lchild_id  ,sycl::read_only};
            sycl::host_accessor old_tree_rchild_id   {*buf_rchild_id  ,sycl::read_only};
            sycl::host_accessor old_tree_lchild_flag {*buf_lchild_flag,sycl::read_only};
            sycl::host_accessor old_tree_rchild_flag {*buf_rchild_flag,sycl::read_only};

            u32 old_tree_leaf_offset = tree_internal_count;

            
            for(u32 i = 0 ; i < ret.tree_leaf_count + ret.tree_internal_count; i++){

                //logger::raw_ln();

                vec3i cur_pos_min_cell_a = new_tree_acc_pos_min_cell[i];
                vec3i cur_pos_max_cell_a = new_tree_acc_pos_max_cell[i];

                u32 cur_id = 0;
                vec3i cur_pos_min_cell_b = old_tree_acc_pos_min_cell[cur_id];
                vec3i cur_pos_max_cell_b = old_tree_acc_pos_max_cell[cur_id];

                while(true){

                    //logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr ->",cur_pos_min_cell_b,cur_pos_max_cell_b);

                    auto is_same_box = [&]() -> bool{
                        return 
                            (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x()) && 
                            (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y()) && 
                            (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z()) && 
                            (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x()) && 
                            (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y()) && 
                            (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z()) ; 
                    };

                    auto potential_cell = [&](vec3i other_min, vec3i other_max) -> bool {
                        return 
                            (cur_pos_min_cell_a.x() >= other_min.x()) && 
                            (cur_pos_min_cell_a.y() >= other_min.y()) && 
                            (cur_pos_min_cell_a.z() >= other_min.z()) && 
                            (cur_pos_max_cell_a.x() <= other_max.x()) && 
                            (cur_pos_max_cell_a.y() <= other_max.y()) && 
                            (cur_pos_max_cell_a.z() <= other_max.z()) ; 
                    };

                    if(is_same_box()){

                        //logger::raw_ln("id : ",i,"found ",cur_id);

                        u32 store_val = cur_id;

                        if(store_val >= tree_internal_count){
                            store_val -= tree_internal_count;
                        }

                        acc_new_node_id_to_old[i] = store_val;

                        break;
                    }


                    u32 lid = old_tree_lchild_id[cur_id] + old_tree_leaf_offset * old_tree_lchild_flag[cur_id];
                    u32 rid = old_tree_rchild_id[cur_id] + old_tree_leaf_offset * old_tree_rchild_flag[cur_id];

                    vec3i cur_pos_min_cell_bl = old_tree_acc_pos_min_cell[lid];
                    vec3i cur_pos_max_cell_bl = old_tree_acc_pos_max_cell[lid];

                    vec3i cur_pos_min_cell_br = old_tree_acc_pos_min_cell[rid];
                    vec3i cur_pos_max_cell_br = old_tree_acc_pos_max_cell[rid];

                    bool l_ok = potential_cell(cur_pos_min_cell_bl,cur_pos_max_cell_bl);
                    bool r_ok = potential_cell(cur_pos_min_cell_br,cur_pos_max_cell_br);

                    //logger::raw_ln("options l=",lid,cur_pos_min_cell_bl,cur_pos_max_cell_bl,l_ok);
                    //logger::raw_ln("options r=",rid,cur_pos_min_cell_br,cur_pos_max_cell_br,r_ok);

                    if(l_ok){

                        cur_pos_min_cell_b = cur_pos_min_cell_bl;
                        cur_pos_max_cell_b = cur_pos_max_cell_bl;

                        cur_id = lid;
                        //logger::raw_ln("id : ",i,"moving to ",cur_id);

                    }else if(r_ok){
                        cur_pos_min_cell_b = cur_pos_min_cell_br;
                        cur_pos_max_cell_b = cur_pos_max_cell_br;

                        cur_id = rid;
                        //logger::raw_ln("id : ",i,"moving to ",cur_id);
                        
                    }else{
                        throw "";
                    }


                    



                }
                
            }
        }

        ret.print_tree_field(*new_node_id_to_old_v1);

        #endif



        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_v2 = std::make_unique<sycl::buffer<u32>>(ret.tree_leaf_count + ret.tree_internal_count);

        {
            auto & new_node_id_to_old = new_node_id_to_old_v2;

            //junk fill
            {
                sycl::host_accessor acc{* new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }



            sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

                
                sycl::accessor acc_new_node_id_to_old {*new_node_id_to_old,cgh,sycl::write_only, sycl::no_init};
                
                sycl::accessor new_tree_acc_pos_min_cell{*ret.buf_pos_min_cell,cgh,sycl::read_only};
                sycl::accessor new_tree_acc_pos_max_cell{*ret.buf_pos_max_cell,cgh,sycl::read_only};

                sycl::accessor old_tree_acc_pos_min_cell{*buf_pos_min_cell,cgh,sycl::read_only};
                sycl::accessor old_tree_acc_pos_max_cell{*buf_pos_max_cell,cgh,sycl::read_only};

                sycl::accessor old_tree_lchild_id   {*buf_lchild_id  ,cgh,sycl::read_only};
                sycl::accessor old_tree_rchild_id   {*buf_rchild_id  ,cgh,sycl::read_only};
                sycl::accessor old_tree_lchild_flag {*buf_lchild_flag,cgh,sycl::read_only};
                sycl::accessor old_tree_rchild_flag {*buf_rchild_flag,cgh,sycl::read_only};

                u32 old_tree_leaf_offset = tree_internal_count;

                sycl::range<1> range_node = sycl::range<1>{ret.tree_leaf_count + ret.tree_internal_count};


                cgh.parallel_for(range_node, [=](sycl::item<1> item) {

                    //logger::raw_ln();

                    vec3i cur_pos_min_cell_a = new_tree_acc_pos_min_cell[item];
                    vec3i cur_pos_max_cell_a = new_tree_acc_pos_max_cell[item];

                    u32 cur_id = 0;
                    vec3i cur_pos_min_cell_b = old_tree_acc_pos_min_cell[cur_id];
                    vec3i cur_pos_max_cell_b = old_tree_acc_pos_max_cell[cur_id];

                    while(true){

                        //logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr ->",cur_pos_min_cell_b,cur_pos_max_cell_b);

                        auto is_same_box = [&]() -> bool{
                            return 
                                (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x()) && 
                                (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y()) && 
                                (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z()) && 
                                (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x()) && 
                                (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y()) && 
                                (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z()) ; 
                        };

                        auto potential_cell = [&](vec3i other_min, vec3i other_max) -> bool {
                            return 
                                (cur_pos_min_cell_a.x() >= other_min.x()) && 
                                (cur_pos_min_cell_a.y() >= other_min.y()) && 
                                (cur_pos_min_cell_a.z() >= other_min.z()) && 
                                (cur_pos_max_cell_a.x() <= other_max.x()) && 
                                (cur_pos_max_cell_a.y() <= other_max.y()) && 
                                (cur_pos_max_cell_a.z() <= other_max.z()) ; 
                        };

                        if(is_same_box()){

                            //logger::raw_ln("id : ",i,"found ",cur_id);

                            u32 store_val = cur_id;

                            if(store_val >= old_tree_leaf_offset){
                                store_val -= old_tree_leaf_offset;
                            }

                            acc_new_node_id_to_old[item] = store_val;

                            break;
                        }


                        u32 lid = old_tree_lchild_id[cur_id] + old_tree_leaf_offset * old_tree_lchild_flag[cur_id];
                        u32 rid = old_tree_rchild_id[cur_id] + old_tree_leaf_offset * old_tree_rchild_flag[cur_id];

                        vec3i cur_pos_min_cell_bl = old_tree_acc_pos_min_cell[lid];
                        vec3i cur_pos_max_cell_bl = old_tree_acc_pos_max_cell[lid];

                        vec3i cur_pos_min_cell_br = old_tree_acc_pos_min_cell[rid];
                        vec3i cur_pos_max_cell_br = old_tree_acc_pos_max_cell[rid];

                        bool l_ok = potential_cell(cur_pos_min_cell_bl,cur_pos_max_cell_bl);
                        bool r_ok = potential_cell(cur_pos_min_cell_br,cur_pos_max_cell_br);

                        //logger::raw_ln("options l=",lid,cur_pos_min_cell_bl,cur_pos_max_cell_bl,l_ok);
                        //logger::raw_ln("options r=",rid,cur_pos_min_cell_br,cur_pos_max_cell_br,r_ok);

                        if(l_ok){

                            cur_pos_min_cell_b = cur_pos_min_cell_bl;
                            cur_pos_max_cell_b = cur_pos_max_cell_bl;

                            cur_id = lid;
                            //logger::raw_ln("id : ",i,"moving to ",cur_id);

                        }else if(r_ok){
                            cur_pos_min_cell_b = cur_pos_min_cell_br;
                            cur_pos_max_cell_b = cur_pos_max_cell_br;

                            cur_id = rid;
                            //logger::raw_ln("id : ",i,"moving to ",cur_id);
                            
                        }                



                    }
                    
                });

            });
        }

        ret.print_tree_field(*new_node_id_to_old_v2);


        
        PatchData ret_pdat(pdat_source.pdl);

        pdat_source.append_subset_to(extract_id,ret_pdat);


        return {std::move(ret), std::move(new_node_id_to_old_v2), std::move(ret_pdat)};

    }
}















template class Radix_Tree<u32, f32_3>;
template class Radix_Tree<u64, f32_3>;

template class Radix_Tree<u32, f64_3>;
template class Radix_Tree<u64, f64_3>;
