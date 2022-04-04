#pragma once


#include "aliases.hpp"
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "kernels/morton_kernels.hpp"
#include "tree/kernels/key_morton_sort.hpp"
#include "tree/kernels/reduction_alg.hpp"


template<class morton_repr>
struct morton_types;

template<>
struct morton_types<u32>{
    using int_repr = u16_3;
};

template<>
struct morton_types<u64>{
    using int_repr = u32_3;
};

inline u32 get_next_pow2_val(u32 val){
    u32 val_rounded_pow = pow(2,32-__builtin_clz(val));
    if(val == pow(2,32-__builtin_clz(val)-1)){
        val_rounded_pow = val;
    }
    return val_rounded_pow;
}


template<class u_morton,class vec3>
class Radix_Tree{public:

    typedef typename morton_types<u_morton>::int_repr vec3i;

    //std::unique_ptr<sycl::buffer<vec3i>> pos_min_buf;

    std::tuple<vec3,vec3> box_coord;
    std::unique_ptr<sycl::buffer<u_morton>> buf_morton;
    std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;


    //aka ranges of index to use
    u32 tree_leaf_count;
    std::vector<u32> reduc_index_map;


    inline void pos_to_buffer(sycl::queue & queue,std::unique_ptr<sycl::buffer<vec3>> & pos_buf){

        u32 morton_len = get_next_pow2_val(pos_buf->size());

        buf_morton = std::make_unique<sycl::buffer<u_morton>>(morton_len);

        sycl_xyz_to_morton<u_morton,vec3>(queue, pos_buf->size(), pos_buf,std::get<0>(box_coord),std::get<1>(box_coord),buf_morton);

        sycl_fill_trailling_buffer<u_morton>(queue, pos_buf->size(),morton_len,buf_morton);
    }

    inline void sort_morton_buf(sycl::queue &queue){
        buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(sycl::range(buf_morton->size()));

        sycl_sort_morton_key_pair(queue, buf_morton->size(), buf_particle_index_map, buf_morton);
    }

    inline Radix_Tree(sycl::queue & queue,std::unique_ptr<sycl::buffer<vec3>> & pos_buf){
        if(pos_buf->size() > i32_max-1){
            throw std::runtime_error("number of element in patch above i32_max-1");
        }

        pos_to_buffer(queue,pos_buf);

        sort_morton_buf(queue);


        // return a sycl buffer from reduc index map instead
        reduction_alg(queue, pos_buf->size(), buf_morton, 0, reduc_index_map, tree_leaf_count);
        

    }



};