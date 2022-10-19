// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "compute_ranges.hpp"

#ifdef SYCL_COMP_DPCPP
#define CLZ(x) sycl::clz(x)
#endif

#ifdef SYCL_COMP_HIPSYCL
#define CLZ_host(x) __hipsycl_if_target_host(__builtin_clz(x))
#define CLZ_cuda(x) __hipsycl_if_target_cuda(__clz(x))
#define CLZ_hip(x) __hipsycl_if_target_hip(__clz(x))
#define CLZ_spirv(x) __hipsycl_if_target_spirv(__clz(x))
#define CLZ(x) CLZ_host(x) CLZ_cuda(x)
#endif


template<>
void sycl_compute_cell_ranges(

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
    std::unique_ptr<sycl::buffer<u16_3>>  & buf_pos_max_cell){


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
        cgh.parallel_for<class ComputeCellRange32>(
            range_radix_tree, [=](sycl::item<1> item) {

                u32 gid =(u32) item.get_id(0);

                uint clz_ = CLZ(morton_map[gid]^morton_map[end_range_map[gid]]);

                
                pos_min_cell[gid] = morton_3d::morton_to_ipos<u32>(morton_map[gid]& (0xFFFFFFFF << (32-clz_)));
                

                pos_max_cell[gid] = morton_3d::get_offset<u32>(clz_) + pos_min_cell[gid];


                if(rchild_flag[gid]){ 
        
                    u16_3 tmp = morton_3d::get_offset<u32>(clz_) - morton_3d::get_offset<u32>(clz_+1);
                    
                    pos_min_cell[rchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid] + tmp;
                    pos_max_cell[rchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u32>(clz_+1) + pos_min_cell[gid] + tmp;
                }
                
                if(lchild_flag[gid]){
                    pos_min_cell[lchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid];
                    pos_max_cell[lchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u32>(clz_+1) + pos_min_cell[gid];
                }



            }
        );


    };

    queue.submit(ker_compute_cell_ranges);

}

template<>
void sycl_compute_cell_ranges(

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
    std::unique_ptr<sycl::buffer<u32_3>>  & buf_pos_max_cell){

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
        cgh.parallel_for<class ComputeCellRange64>(
            range_radix_tree, [=](sycl::item<1> item) {

                u32 gid =(u32) item.get_id(0);

                uint clz_ = CLZ(morton_map[gid]^morton_map[end_range_map[gid]]);

                #if defined(PRECISION_MORTON_DOUBLE)
                    pos_min_cell[gid] = morton_to_ixyz(morton_map[gid]& (0xFFFFFFFFFFFFFFFF << (64-clz_)));
                #else
                    pos_min_cell[gid] = morton_3d::morton_to_ipos<u64>(morton_map[gid]& (0xFFFFFFFF << (32-clz_)));
                #endif

                pos_max_cell[gid] = morton_3d::get_offset<u64>(clz_) + pos_min_cell[gid];


                if(rchild_flag[gid]){ 
        
                    u32_3 tmp = morton_3d::get_offset<u64>(clz_) - morton_3d::get_offset<u64>(clz_+1);
                    
                    pos_min_cell[rchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid] + tmp;
                    pos_max_cell[rchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u64>(clz_+1) + pos_min_cell[gid] + tmp;
                }
                
                if(lchild_flag[gid]){
                    pos_min_cell[lchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid];
                    pos_max_cell[lchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u64>(clz_+1) + pos_min_cell[gid];
                }



            }
        );


    };

    queue.submit(ker_compute_cell_ranges);
}


#if false


template <class u_morton>
inline typename morton_3d::morton_types<u_morton>::int_vec_repr get_pmin_ipos(const u_morton &m_code, const uint &clz_val);


template<> 
inline u16_3 get_pmin_ipos(const u32 &m_code, const uint &clz_val){
    return morton_3d::morton_to_ipos<u32>(m_code & (0xFFFFFFFF << (32-clz_val)));
}

template<> 
inline u32_3 get_pmin_ipos(const u64 &m_code, const uint &clz_val){
    return morton_3d::morton_to_ipos<u64>(m_code & (0xFFFFFFFFFFFFFFFF << (64-clz_val)));
}





template<class u_morton>
class ComputeCellRange;





template<class u_morton>
inline void internal_sycl_compute_cell_ranges(

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
        cgh.parallel_for<ComputeCellRange<u_morton>>(
            range_radix_tree, [=](sycl::item<1> item) {

                u32 gid =(u32) item.get_id(0);

                u_morton morton_gid = morton_map[gid];

                uint clz_ = CLZ(morton_gid^morton_map[end_range_map[gid]]);

                u8 rflag = rchild_flag[gid];
                u8 lflag = lchild_flag[gid];

                auto offset_clz = morton_3d::get_offset<u_morton>(clz_);

                auto pmin_nid = get_pmin_ipos(morton_gid,clz_);
                auto pmax_nid = offset_clz + pos_min_cell[gid];

                pos_min_cell[gid] = pmin_nid;
                pos_max_cell[gid] = pmax_nid;

                if(rflag || lflag){

                    auto offset_clzp1 = morton_3d::get_offset<u_morton>(clz_+1);

                    if(rflag){ 
            
                        auto tmp = offset_clz - offset_clzp1;

                        const u32 rid = rchild_id[gid];
                        
                        pos_min_cell[rid+internal_cell_cnt] = pmin_nid + tmp;
                        pos_max_cell[rid+internal_cell_cnt] = offset_clzp1 + pmin_nid + tmp;
                    }
                    
                    if(lflag){

                        const u32 lid = lchild_id[gid];
                        pos_min_cell[lid+internal_cell_cnt] = pmin_nid;
                        pos_max_cell[lid+internal_cell_cnt] = offset_clzp1 + pmin_nid;
                    }

                }
                



            }
        );


    };

    queue.submit(ker_compute_cell_ranges);

}















template <>
    void sycl_compute_cell_ranges(

        sycl::queue &queue,

        u32 leaf_cnt, u32 internal_cnt, std::unique_ptr<sycl::buffer<u32>> &buf_morton,
        std::unique_ptr<sycl::buffer<u32>> &buf_lchild_id, std::unique_ptr<sycl::buffer<u32>> &buf_rchild_id,
        std::unique_ptr<sycl::buffer<u8>> &buf_lchild_flag, std::unique_ptr<sycl::buffer<u8>> &buf_rchild_flag,
        std::unique_ptr<sycl::buffer<u32>> &buf_endrange,

        std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_min_cell, std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_max_cell
    ) {

    internal_sycl_compute_cell_ranges(queue, leaf_cnt, internal_cnt, buf_morton, buf_lchild_id, buf_rchild_id, buf_lchild_flag, buf_rchild_flag, buf_endrange, buf_pos_min_cell, buf_pos_max_cell);

}

//pos_min_cell[gid] = morton_3d::morton_to_ipos<u64>(morton_map[gid]& (0xFFFFFFFFFFFFFFFF << (64-clz_)));
template<>
void sycl_compute_cell_ranges(

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
    std::unique_ptr<sycl::buffer<u32_3>>  & buf_pos_max_cell){

    internal_sycl_compute_cell_ranges(queue, leaf_cnt, internal_cnt, buf_morton, buf_lchild_id, buf_rchild_id, buf_lchild_flag, buf_rchild_flag, buf_endrange, buf_pos_min_cell, buf_pos_max_cell);
}


#endif