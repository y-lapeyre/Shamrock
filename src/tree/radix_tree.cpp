#include "radix_tree.hpp"

#include "CL/sycl/buffer.hpp"
#include "kernels/karras_alg.hpp"
#include "kernels/key_morton_sort.hpp"
#include "kernels/reduction_alg.hpp"

void Radix_Tree::build_tree(
        sycl::queue* queue,
        sycl::buffer<u_morton>* buf_morton, 
        u32 morton_code_count, 
        u32 morton_code_count_rounded_pow,
        bool use_reduction, 
        u32 reduction_level){

    this->is_reduction_active = use_reduction;

    sort_index_map = new sycl::buffer<u32>(morton_code_count_rounded_pow);

    sycl_sort_morton_key_pair(queue, morton_code_count_rounded_pow, sort_index_map, buf_morton);

    if(!use_reduction){

        leaf_cell_count= morton_code_count;
        buf_leaf_morton = buf_morton;

    }else{

        reduc_index_map.clear();
        leaf_cell_count = 0;

        reduction_alg(
            //in
            queue,
            morton_code_count,
            buf_morton,
            reduction_level,
            //out
            reduc_index_map,
            leaf_cell_count);

        buf_reduc_index_map = new sycl::buffer<u32     >( reduc_index_map );
        buf_leaf_morton     = new sycl::buffer<u_morton>(leaf_cell_count);

        sycl_morton_remap_reduction(queue, leaf_cell_count, buf_reduc_index_map, buf_morton, buf_leaf_morton);

        //TODO store reduction factor

    }

    internal_cell_count = leaf_cell_count -1;

    //TODO add check leaf cell count ==1

    buf_rchild_id   = new sycl::buffer<u32>(internal_cell_count);
    buf_lchild_id   = new sycl::buffer<u32>(internal_cell_count);
    buf_rchild_flag = new sycl::buffer<u8> (internal_cell_count);
    buf_lchild_flag = new sycl::buffer<u8> (internal_cell_count);
    buf_endrange    = new sycl::buffer<u32>(internal_cell_count);

    sycl_karras_alg(queue, 
        internal_cell_count, 
        buf_leaf_morton, 
        buf_rchild_id,
        buf_lchild_id,
        buf_rchild_flag,
        buf_lchild_flag,
        buf_endrange);

    throw_with_pos("need to implement bos dimension buffer computation")

}