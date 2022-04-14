#pragma once


#include "aliases.hpp"
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "kernels/morton_kernels.hpp"
#include "sfc/morton.hpp"
#include "tree/kernels/compute_ranges.hpp"
#include "tree/kernels/convert_ranges.hpp"
#include "tree/kernels/karras_alg.hpp"
#include "tree/kernels/key_morton_sort.hpp"
#include "tree/kernels/reduction_alg.hpp"
#include "utils/geometry_utils.hpp"




inline u32 get_next_pow2_val(u32 val){
    u32 val_rounded_pow = pow(2,32-__builtin_clz(val));
    if(val == pow(2,32-__builtin_clz(val)-1)){
        val_rounded_pow = val;
    }
    return val_rounded_pow;
}

template<class u_morton>
class Radix_tree_depth;

template<>
class Radix_tree_depth<u32>{public:
    static constexpr u32 tree_depth = 32;
};

template<>
class Radix_tree_depth<u64>{public:
    static constexpr u32 tree_depth = 64;
};


template<class u_morton,class vec3>
class Radix_Tree{public:

    typedef typename morton_3d::morton_types<u_morton>::int_vec_repr vec3i;
    typedef typename vec3::element_type flt;

    static constexpr u32 tree_depth = Radix_tree_depth<u_morton>::tree_depth;

    //std::unique_ptr<sycl::buffer<vec3i>> pos_min_buf;

    std::tuple<vec3,vec3> box_coord;
    std::unique_ptr<sycl::buffer<u_morton>> buf_morton;
    std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;


    //aka ranges of index to use
    u32 tree_leaf_count;
    std::vector<u32> reduc_index_map;
    std::unique_ptr<sycl::buffer<u32>> buf_reduc_index_map;

    bool one_cell_mode = false;

    u32 tree_internal_count;
    std::unique_ptr<sycl::buffer<u_morton>> buf_tree_morton;
    std::unique_ptr<sycl::buffer<u32>>      buf_lchild_id;
    std::unique_ptr<sycl::buffer<u32>>      buf_rchild_id;
    std::unique_ptr<sycl::buffer<u8>>       buf_lchild_flag;
    std::unique_ptr<sycl::buffer<u8>>       buf_rchild_flag;
    std::unique_ptr<sycl::buffer<u32>>      buf_endrange;

    inline Radix_Tree(sycl::queue & queue,std::tuple<vec3,vec3> treebox,std::unique_ptr<sycl::buffer<vec3>> & pos_buf){
        if(pos_buf->size() > i32_max-1){
            throw std::runtime_error("number of element in patch above i32_max-1");
        }


        std::cout 
            << std::get<0>(treebox).x() <<","<<std::get<0>(treebox).y() <<","<<std::get<0>(treebox).z() << " | "
            << std::get<1>(treebox).x() <<","<<std::get<1>(treebox).y() <<","<<std::get<1>(treebox).z() << std::endl;

        box_coord = treebox;

        std::cout << "pos_to_morton" << std::endl;
        u32 morton_len = get_next_pow2_val(pos_buf->size());

        buf_morton = std::make_unique<sycl::buffer<u_morton>>(morton_len);

        sycl_xyz_to_morton<u_morton,vec3>(queue, pos_buf->size(), pos_buf,std::get<0>(box_coord),std::get<1>(box_coord),buf_morton);

        sycl_fill_trailling_buffer<u_morton>(queue, pos_buf->size(),morton_len,buf_morton);


        std::cout << "sort_morton_buf" << std::endl;
        buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(buf_morton->size());

        queue.submit([&](sycl::handler &cgh) {
            auto pidm = buf_particle_index_map->get_access<sycl::access::mode::discard_write>(cgh);
            cgh.parallel_for(sycl::range(buf_morton->size()), [=](sycl::item<1> item) {
                pidm[item] = item.get_id(0);
            });
        });

        sycl_sort_morton_key_pair(queue, buf_morton->size(), buf_particle_index_map, buf_morton);


        // return a sycl buffer from reduc index map instead
        std::cout << "reduction_alg" << std::endl;
        reduction_alg(queue, pos_buf->size(), buf_morton, 0, reduc_index_map, tree_leaf_count);
        
        std::cout << " -> " << pos_buf->size() << " " << buf_morton->size() << " " << tree_leaf_count << std::endl;


        buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(reduc_index_map.data(),reduc_index_map.size());

        std::cout << "sycl_morton_remap_reduction" << std::endl;
        buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(tree_leaf_count);

        sycl_morton_remap_reduction(queue, tree_leaf_count, buf_reduc_index_map, buf_morton, buf_tree_morton);



        if(tree_leaf_count > 3){

            tree_internal_count = tree_leaf_count-1;

            buf_lchild_id = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
            buf_rchild_id = std::make_unique<sycl::buffer<u32>>(tree_internal_count);
            buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
            buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(tree_internal_count);
            buf_endrange = std::make_unique<sycl::buffer<u32>>(tree_internal_count);

            sycl_karras_alg(queue, tree_internal_count, buf_tree_morton, buf_lchild_id, buf_rchild_id, buf_lchild_flag, buf_rchild_flag, buf_endrange);

            one_cell_mode = false;
        }else{
            one_cell_mode = true;
        }
    }

    std::unique_ptr<sycl::buffer<vec3i>> buf_pos_min_cell;
    std::unique_ptr<sycl::buffer<vec3i>> buf_pos_max_cell;

    std::unique_ptr<sycl::buffer<vec3>> buf_pos_min_cell_flt;
    std::unique_ptr<sycl::buffer<vec3>> buf_pos_max_cell_flt;

    inline void compute_cellvolume(sycl::queue & queue){

        std::cout << "compute_cellvolume" << std::endl;

        
        buf_pos_min_cell = std::make_unique< sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);
        buf_pos_max_cell = std::make_unique< sycl::buffer<vec3i>>(tree_internal_count + tree_leaf_count);


        sycl_compute_cell_ranges(
            queue, 
            tree_leaf_count,
            tree_internal_count, 
            buf_tree_morton, 
            buf_lchild_id, 
            buf_rchild_id, 
            buf_lchild_flag, 
            buf_rchild_flag, 
            buf_endrange, buf_pos_min_cell, buf_pos_max_cell);


        buf_pos_min_cell_flt = std::make_unique< sycl::buffer<vec3>>(tree_internal_count + tree_leaf_count);
        buf_pos_max_cell_flt = std::make_unique< sycl::buffer<vec3>>(tree_internal_count + tree_leaf_count);

        std::cout << "sycl_convert_cell_range" << std::endl;

        sycl_convert_cell_range<u_morton,vec3>(queue, tree_leaf_count, tree_internal_count, std::get<0>(box_coord), std::get<1>(box_coord), 
            buf_pos_min_cell, 
            buf_pos_max_cell, 
            buf_pos_min_cell_flt, 
            buf_pos_max_cell_flt);

    }


    std::unique_ptr<sycl::buffer<flt>> buf_cell_interact_rad;


    inline void compute_int_boxes(sycl::queue & queue,std::unique_ptr<sycl::buffer<flt>> & int_rad_buf){

        buf_cell_interact_rad = std::make_unique< sycl::buffer<flt>>(tree_internal_count + tree_leaf_count);
        sycl::range<1> range_leaf_cell{tree_leaf_count};


        queue.submit([&](sycl::handler &cgh) {

            u32 offset_leaf = tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::discard_write>(cgh);
            auto h = int_rad_buf->template get_access<sycl::access::mode::read>(cgh);

            auto cell_particle_ids = buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
            auto particle_index_map = buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);


            cgh.parallel_for(
                range_leaf_cell, [=](sycl::item<1> item) {
                    u32 gid = (u32) item.get_id(0);

                    u32 min_ids = cell_particle_ids[gid];
                    u32 max_ids = cell_particle_ids[gid+1];
                    f32 h_tmp = 0;

                    for(unsigned int id_s = min_ids; id_s < max_ids;id_s ++){
        
                        f32 h_a = h[particle_index_map[id_s]];
                        h_tmp = (h_tmp > h_a ? h_tmp : h_a);
                        
                    }
                    
                    h_max_cell[offset_leaf + gid] = h_tmp;

                }
            );


        });



        sycl::range<1> range_tree{tree_internal_count};
        auto ker_reduc_hmax = [&](sycl::handler &cgh) {

            u32 offset_leaf = tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::read_write>(cgh);

            auto rchild_id      = buf_rchild_id  ->get_access<sycl::access::mode::read>(cgh);
            auto lchild_id      = buf_lchild_id  ->get_access<sycl::access::mode::read>(cgh);
            auto rchild_flag    = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
            auto lchild_flag    = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(
                range_tree, [=](sycl::item<1> item) {

                    u32 gid = (u32) item.get_id(0);

                    u32 lid = lchild_id[gid] + offset_leaf*lchild_flag[gid];
                    u32 rid = rchild_id[gid] + offset_leaf*rchild_flag[gid];
                    
                    flt h_l = h_max_cell[lid];
                    flt h_r = h_max_cell[rid];
                    
                    h_max_cell[gid] = (h_r > h_l ? h_r : h_l);

                }
            );

        };

        
        for(u32 i = 0 ; i < tree_depth ; i++){
            queue.submit(ker_reduc_hmax);
        }


    }



};


namespace walker {

    namespace interaction_crit {
        template<class vec3,class flt>
        inline bool sph_radix_cell_crit(vec3 xyz_a,vec3 part_a_box_min,vec3 part_a_box_max,vec3 cur_cell_box_min,vec3 cur_cell_box_max,flt box_int_sz){
            

            vec3 inter_box_b_min = cur_cell_box_min - box_int_sz;
            vec3 inter_box_b_max = cur_cell_box_max + box_int_sz;

            return 
                BBAA::cella_neigh_b(
                    part_a_box_min, part_a_box_max, 
                    cur_cell_box_min, cur_cell_box_max) ||
                BBAA::cella_neigh_b(
                    xyz_a, xyz_a,                   
                    inter_box_b_min, inter_box_b_max);
        }
    }


    

    template<class u_morton,class vec3>
    class Radix_tree_accessor{public:
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  particle_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device> cell_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  rchild_id     ;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  lchild_id     ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  rchild_flag   ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  lchild_flag   ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_min_cell  ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_max_cell  ;

        static constexpr u32 tree_depth = Radix_tree_depth<u_morton>::tree_depth;
        static constexpr u32 _nindex = 4294967295;

        u32 leaf_offset;

        
        Radix_tree_accessor(Radix_Tree< u_morton,  vec3> & rtree,sycl::handler & cgh):
            particle_index_map(rtree.buf_particle_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            cell_index_map(rtree.buf_reduc_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            rchild_id     (rtree.buf_rchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            lchild_id     (rtree.buf_lchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            rchild_flag   (rtree.buf_rchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            lchild_flag   (rtree.buf_lchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            pos_min_cell  (rtree.buf_pos_min_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            pos_max_cell  (rtree.buf_pos_max_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            leaf_offset   (rtree.tree_internal_count)
        {}
    };

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
    inline void rtree_for(Rta &acc, Functor_int_cd &&func_int_cd, Functor_iter &&func_it, Functor_iter_excl &&func_excl) {
        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        while (stack_cursor < Rta::tree_depth) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and can interact => force
                if (current_node_id >= acc.leaf_offset) {

                    // loop on particle indexes
                    uint min_ids = acc.cell_index_map[current_node_id     -acc.leaf_offset];
                    uint max_ids = acc.cell_index_map[current_node_id + 1 -acc.leaf_offset];

                    for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                        //recover old index before morton sort
                        uint id_b = acc.particle_index_map[id_s];

                        //iteration function
                        func_it(id_b);
                    }

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                func_excl(current_node_id);
            }
        }
    }



}