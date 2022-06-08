#include "CL/sycl/buffer.hpp"
#include "CL/sycl/range.hpp"
#include "aliases.hpp"
#include "utils/sycl_vector_utils.hpp"
#include <iostream>

template<class T, class flt>
inline void field_advance_time(sycl::queue & queue, sycl::buffer<T> & buf_val, sycl::buffer<T> & buf_der, sycl::range<1> elem_range, flt dt){

    auto ker_advance_time = [&](sycl::handler &cgh) {
        auto acc_du = buf_val.template get_access<sycl::access::mode::read>(cgh);
        auto acc_u = buf_der.template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            T du = acc_du[item];

            acc_u[item] = acc_u[item] + (dt) * (du);

        });
    };




    queue.submit(ker_advance_time);




    std::cout << "res  ###############" << std::endl;
    {
        auto acc_du = buf_val.template get_access<sycl::access::mode::read>();
        auto acc_u = buf_der.template get_access<sycl::access::mode::read>();

        std::cout << "v: ";
        print_vec(std::cout, acc_u[1000]);
        std::cout << " a: ";
        print_vec(std::cout, acc_du[1000]);
        std::cout << std::endl;

    }
}