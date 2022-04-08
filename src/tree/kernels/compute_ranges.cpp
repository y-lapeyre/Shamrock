
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

        auto pos_min_cell = buf_pos_min_cell->template get_access<sycl::access::mode::write>(cgh); 
        auto pos_max_cell = buf_pos_max_cell->template get_access<sycl::access::mode::write>(cgh); 

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

                
                pos_min_cell[gid] = morton_3d::morton_to_ipos<u32,f32>(morton_map[gid]& (0xFFFFFFFF << (32-clz_)));
                

                pos_max_cell[gid] = morton_3d::get_offset<u32, f32>(clz_) + pos_min_cell[gid];


                if(rchild_flag[gid]){ 
        
                    u16_3 tmp = morton_3d::get_offset<u32, f32>(clz_) - morton_3d::get_offset<u32, f32>(clz_+1);
                    
                    pos_min_cell[rchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid] + tmp;
                    pos_max_cell[rchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u32, f32>(clz_+1) + pos_min_cell[gid] + tmp;
                }
                
                if(lchild_flag[gid]){
                    pos_min_cell[lchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid];
                    pos_max_cell[lchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u32, f32>(clz_+1) + pos_min_cell[gid];
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

        auto pos_min_cell = buf_pos_min_cell->template get_access<sycl::access::mode::write>(cgh); 
        auto pos_max_cell = buf_pos_max_cell->template get_access<sycl::access::mode::write>(cgh); 

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
                    pos_min_cell[gid] = morton_3d::morton_to_ipos<u64,f32>(morton_map[gid]& (0xFFFFFFFF << (32-clz_)));
                #endif

                pos_max_cell[gid] = morton_3d::get_offset<u64, f32>(clz_) + pos_min_cell[gid];


                if(rchild_flag[gid]){ 
        
                    u32_3 tmp = morton_3d::get_offset<u64, f32>(clz_) - morton_3d::get_offset<u64, f32>(clz_+1);
                    
                    pos_min_cell[rchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid] + tmp;
                    pos_max_cell[rchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u64, f32>(clz_+1) + pos_min_cell[gid] + tmp;
                }
                
                if(lchild_flag[gid]){
                    pos_min_cell[lchild_id[gid]+internal_cell_cnt] = pos_min_cell[gid];
                    pos_max_cell[lchild_id[gid]+internal_cell_cnt] = morton_3d::get_offset<u64, f32>(clz_+1) + pos_min_cell[gid];
                }



            }
        );


    };

    queue.submit(ker_compute_cell_ranges);
}