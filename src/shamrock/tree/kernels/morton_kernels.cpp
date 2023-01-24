// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "morton_kernels.hpp"
#include "aliases.hpp"
#include "shamrock/sfc/morton.hpp"
#include <memory>


using namespace shamrock::sfc;

template<>
void sycl_xyz_to_morton<u32,f32_3,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> & in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> & out_morton){

    sycl::range<1> range_cnt{pos_count};

    using Morton = shamrock::sfc::MortonCodes<u32, 3>;

    queue.submit(
        [&](sycl::handler &cgh) {

            f32_3 b_box_min = bounding_box_min;
            f32_3 b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f32_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = Morton::coord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u32,f64_3,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> & in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    using Morton = shamrock::sfc::MortonCodes<u32, 3>;

    queue.submit(
        [&](sycl::handler &cgh) {

            f64_3 b_box_min = bounding_box_min;
            f64_3 b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f64_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = Morton::coord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u64,f32_3,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> & in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> & out_morton){

    sycl::range<1> range_cnt{pos_count};

    
    using Morton = shamrock::sfc::MortonCodes<u64, 3>;

    queue.submit(
        [&](sycl::handler &cgh) {

            f32_3 b_box_min = bounding_box_min;
            f32_3 b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f32_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = Morton::coord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u64,f64_3,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> & in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    using Morton = shamrock::sfc::MortonCodes<u64, 3>;

    queue.submit(
        [&](sycl::handler &cgh) {

            f64_3 b_box_min = bounding_box_min;
            f64_3 b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f64_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = Morton::coord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}








template<>
void sycl_xyz_to_morton<u32,MortonCodes<u32, 3>::int_vec_repr,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<MortonCodes<u32, 3>::int_vec_repr>> & in_positions,
    MortonCodes<u32, 3>::int_vec_repr bounding_box_min,
    MortonCodes<u32, 3>::int_vec_repr bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> & out_morton){

    using Morton = shamrock::sfc::MortonCodes<u32, 3>;
    using pos_t = MortonCodes<u32, 3>::int_vec_repr;

    if (!Morton::is_morton_bounding_box(bounding_box_min, bounding_box_max)) {
        throw std::invalid_argument("integer coordinates can be used with MortonBuilder only if the boundaing box matches the coordinate range of morton codes");
    }

    sycl::range<1> range_cnt{pos_count};

    queue.submit(
        [&](sycl::handler &cgh) {

            pos_t b_box_min = bounding_box_min;
            pos_t b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                pos_t r = xyz[i];

                m[i] = Morton::icoord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u64,MortonCodes<u64, 3>::int_vec_repr,3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<MortonCodes<u64, 3>::int_vec_repr>> & in_positions,
    MortonCodes<u64, 3>::int_vec_repr bounding_box_min,
    MortonCodes<u64, 3>::int_vec_repr bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> & out_morton){

    using Morton = shamrock::sfc::MortonCodes<u64, 3>;
    using pos_t = MortonCodes<u64, 3>::int_vec_repr;

    if (!Morton::is_morton_bounding_box(bounding_box_min, bounding_box_max)) {
        throw std::invalid_argument("integer coordinates can be used with MortonBuilder only if the boundaing box matches the coordinate range of morton codes");
    }

    sycl::range<1> range_cnt{pos_count};

    queue.submit(
        [&](sycl::handler &cgh) {

            pos_t b_box_min = bounding_box_min;
            pos_t b_box_max = bounding_box_max;

            sycl::accessor xyz {*in_positions, cgh, sycl::read_only};
            sycl::accessor m   {*out_morton, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                pos_t r = xyz[i];

                m[i] = Morton::icoord_to_morton(r.x(),r.y(),r.z());
                
            });


        }

    );

}










template<>
void sycl_fill_trailling_buffer<u32>(
    sycl::queue & queue,
    u32 morton_count,
    u32 fill_count,
    std::unique_ptr<sycl::buffer<u32>> & buf_morton
    ){

    if(fill_count - morton_count == 0){
        std::cout << "skipping" << std::endl;
        return;
    }

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class FillTraillingBuf_u32>(
            range_npart, [=](sycl::item<1> i) {
                m[morton_count + i.get_id()] = 4294967295U;
            }
        );

    };

    queue.submit(ker_fill_trailling_buf);


}

template<>
void sycl_fill_trailling_buffer<u64>(
    sycl::queue & queue,
    u32 morton_count,
    u32 fill_count,
    std::unique_ptr<sycl::buffer<u64>> & buf_morton
    ){

    if(fill_count - morton_count == 0){
        std::cout << "skipping" << std::endl;
        return;
    }

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class FillTraillingBuf_u64>(
            range_npart, [=](sycl::item<1> i) {
                m[morton_count + i.get_id()] = 18446744073709551615UL;
            }
        );

    };

    queue.submit(ker_fill_trailling_buf);


}