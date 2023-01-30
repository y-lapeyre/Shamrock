// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "compute_ranges.hpp"

#include "shamrock/math/integerManip.hpp"



template<class u_morton>
void sycl_compute_cell_ranges(

    sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt ,
    std::unique_ptr<sycl::buffer<u_morton>> & buf_morton,
    std::unique_ptr<sycl::buffer<u32>> & buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> & buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>>  & buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>>  & buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> & buf_endrange,
    
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_max_cell){

    using namespace shamrock::math::int_manip;

    sycl::range<1> range_radix_tree{internal_cnt};

    auto ker_compute_cell_ranges = [&](sycl::handler &cgh) {

        auto morton_map = buf_morton->template get_access<sycl::access::mode::read>(cgh); 
        auto end_range_map = buf_endrange->get_access<sycl::access::mode::read>(cgh); 

        auto pos_min_cell = buf_pos_min_cell->template get_access<sycl::access::mode::discard_write>(cgh); //was "write" before changed to fix warning 
        auto pos_max_cell = buf_pos_max_cell->template get_access<sycl::access::mode::discard_write>(cgh); //was "write" before changed to fix warning 

        auto rchild_flag    = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto lchild_flag    = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto rchild_id      = buf_rchild_id  ->get_access<sycl::access::mode::read>(cgh);
        auto lchild_id      = buf_lchild_id  ->get_access<sycl::access::mode::read>(cgh);

        u32 internal_cell_cnt = internal_cnt;

        // Executing kernel
        cgh.parallel_for(
            range_radix_tree, [=](sycl::item<1> item) {

                u32 gid =(u32) item.get_id(0);

                uint clz_ = clz_xor(morton_map[gid],morton_map[end_range_map[gid]]);

                if constexpr (std::is_same<u_morton,u64>::value){
                    pos_min_cell[gid] = morton_3d::morton_to_ipos<u64>(morton_map[gid]& (0xFFFFFFFFFFFFFFFF << (64-clz_)));
                }else{
                    pos_min_cell[gid] = morton_3d::morton_to_ipos<u32>(morton_map[gid]& (0xFFFFFFFF << (32-clz_)));
                }

                pos_max_cell[gid] = morton_3d::get_offset<u_morton>(clz_) + pos_min_cell[gid];


                if(rchild_flag[gid]){ 
        
                    auto tmp = morton_3d::get_offset<u_morton>(clz_) - morton_3d::get_offset<u_morton>(clz_+1);
                    
                    pos_min_cell[rchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid] + tmp;
                    pos_max_cell[rchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u_morton>(clz_+1) + pos_min_cell[gid] + tmp;
                }
                
                if(lchild_flag[gid]){
                    pos_min_cell[lchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid];
                    pos_max_cell[lchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u_morton>(clz_+1) + pos_min_cell[gid];
                }



            }
        );


    };

    queue.submit(ker_compute_cell_ranges);
}


template void sycl_compute_cell_ranges(
    sycl::queue & queue,
    u32 leaf_cnt , 
    u32 internal_cnt ,
    std::unique_ptr<sycl::buffer<u32>> & buf_morton,
    std::unique_ptr<sycl::buffer<u32>> & buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> & buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>>  & buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>>  & buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> & buf_endrange,
    
    std::unique_ptr<sycl::buffer<u16_3>>  & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>>  & buf_pos_max_cell);


template void sycl_compute_cell_ranges(
    sycl::queue & queue,
    u32 leaf_cnt , 
    u32 internal_cnt ,
    std::unique_ptr<sycl::buffer<u64>> & buf_morton,
    std::unique_ptr<sycl::buffer<u32>> & buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> & buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>>  & buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>>  & buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> & buf_endrange,
    
    std::unique_ptr<sycl::buffer<u32_3>>  & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>>  & buf_pos_max_cell);

