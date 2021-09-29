#include "morton.hpp"

namespace morton {

    void sycl_xyz_to_morton(
        sycl::queue* queue,
        u32 pos_count,
        sycl::buffer<f3_d>* in_positions,
        f3_d bounding_box_min,
        f3_d bounding_box_max,
        sycl::buffer<u_morton>* out_morton){

        cl::sycl::range<1> range_cnt{pos_count};
    

        queue->submit(
            [&](cl::sycl::handler &cgh) {

                f3_d b_box_min = bounding_box_min;
                f3_d b_box_max = bounding_box_max;

                auto xyz = in_positions->get_access<sycl::access::mode::read>(cgh);
                auto m   = out_morton  ->get_access<sycl::access::mode::discard_write>(cgh);
                
                cgh.parallel_for<class Kernel_xyz_to_morton>(range_cnt, [=](cl::sycl::item<1> item) {

                    int i = (int) item.get_id(0);
                    
                    f3_d r = xyz[i];

                    r -= b_box_min;
                    r /= b_box_max - b_box_min;

                    m[i] = xyz_to_morton(r.s0(), r.s1(), r.s2());
                    
                });


            }

        );

    }


    void sycl_fill_trailling_buffer(
        sycl::queue* queue,
        u32 morton_count,
        u32 fill_count,
        sycl::buffer<u_morton>* buf_morton
        ){

        cl::sycl::range<1> range_npart{fill_count - morton_count};

        auto ker_fill_trailling_buf = [&](cl::sycl::handler &cgh) {
            
            auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

            // Executing kernel
            cgh.parallel_for<class Kernel_fill_trailling_buffer>(
                range_npart, [=](cl::sycl::item<1> i) {
                    #if defined(PRECISION_MORTON_DOUBLE)
                        m[morton_count + i.get_id()] = 18446744073709551615ul;
                    #else
                        m[morton_count + i.get_id()] = 4294967295u;
                    #endif
                }
            );

        };

        queue->submit(ker_fill_trailling_buf);


    }

}

