#include "morton_kernels.hpp"
#include "aliases.hpp"
#include "sfc/morton.hpp"
#include <memory>



template<>
void sycl_xyz_to_morton<u32,f32_3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> & in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    queue.submit(
        [&](sycl::handler &cgh) {

            f32_3 b_box_min = bounding_box_min;
            f32_3 b_box_max = bounding_box_max;

            auto xyz = in_positions->get_access<sycl::access::mode::read>(cgh);
            auto m   = out_morton  ->get_access<sycl::access::mode::discard_write>(cgh);
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f32_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = morton_3d::coord_to_morton<u32,f32>(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u32,f64_3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> & in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    queue.submit(
        [&](sycl::handler &cgh) {

            f64_3 b_box_min = bounding_box_min;
            f64_3 b_box_max = bounding_box_max;

            auto xyz = in_positions->get_access<sycl::access::mode::read>(cgh);
            auto m   = out_morton  ->get_access<sycl::access::mode::discard_write>(cgh);
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f64_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = morton_3d::coord_to_morton<u32,f64>(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u64,f32_3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> & in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    queue.submit(
        [&](sycl::handler &cgh) {

            f32_3 b_box_min = bounding_box_min;
            f32_3 b_box_max = bounding_box_max;

            auto xyz = in_positions->get_access<sycl::access::mode::read>(cgh);
            auto m   = out_morton  ->get_access<sycl::access::mode::discard_write>(cgh);
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f32_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = morton_3d::coord_to_morton<u64,f32>(r.x(),r.y(),r.z());
                
            });


        }

    );

}

template<>
void sycl_xyz_to_morton<u64,f64_3>(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> & in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> & out_morton){

    sycl::range<1> range_cnt{pos_count};


    queue.submit(
        [&](sycl::handler &cgh) {

            f64_3 b_box_min = bounding_box_min;
            f64_3 b_box_max = bounding_box_max;

            auto xyz = in_positions->get_access<sycl::access::mode::read>(cgh);
            auto m   = out_morton  ->get_access<sycl::access::mode::discard_write>(cgh);
            
            cgh.parallel_for(range_cnt, [=](sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                f64_3 r = xyz[i];

                r -= b_box_min;
                r /= b_box_max - b_box_min;

                m[i] = morton_3d::coord_to_morton<u64,f64>(r.x(),r.y(),r.z());
                
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

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class FillTraillingBuf_u32>(
            range_npart, [=](sycl::item<1> i) {
                m[morton_count + i.get_id()] = 4294967295u;
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

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class FillTraillingBuf_u64>(
            range_npart, [=](sycl::item<1> i) {
                m[morton_count + i.get_id()] = 18446744073709551615ul;
            }
        );

    };

    queue.submit(ker_fill_trailling_buf);


}