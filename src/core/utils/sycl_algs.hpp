// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/sys/sycl_handler.hpp"


//%Impl status : Clean

namespace syclalgs {

    namespace basic {

        template <class T> void copybuf(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template <class T> void copybuf_discard(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template<class T>
        void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count);

    } // namespace basic

    namespace reduction {

        bool is_all_true(sycl::buffer<u8> & buf, u32 cnt);

        template <class T> 
        bool equals(sycl::buffer<T> & buf1, sycl::buffer<T> & buf2, u32 cnt);

        template<class T, class Op, u32 work_group_size = 16> inline T reduce(sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){

            u32 len = end_id - start_id;


            sycl::buffer<T> buf_int(len);
            ::syclalgs::basic::write_with_offset_into(buf_int,buf1,start_id,len);


            u32 part_size = work_group_size * 2;

            while (len != 1) {
                u32 n_wgroups = (len + part_size - 1) / part_size;    

                q.submit([&] (sycl::handler& cgh) {
                    sycl::local_accessor<T> local_mem {sycl::range<1>(work_group_size), cgh};

                    sycl::accessor global_mem {buf_int,cgh, sycl::read_write};
                    
                    cgh.parallel_for(
                        sycl::nd_range<1>(n_wgroups * work_group_size, work_group_size),
                        [=] (sycl::nd_item<1> item) {
                        
                        size_t local_id = item.get_local_linear_id();
                        size_t global_id = item.get_global_linear_id();
                        local_mem[local_id] = 0;

                        if ((2 * global_id) < len) {
                            local_mem[local_id] = Op{}(global_mem[2 * global_id] , global_mem[2 * global_id + 1]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        for (size_t stride = 1; stride < work_group_size; stride *= 2) {
                            auto idx = 2 * stride * local_id;
                            if (idx < work_group_size) {
                                local_mem[idx] = Op{}(local_mem[idx] , local_mem[idx + stride]);
                            }

                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        if (local_id == 0) {
                            global_mem[item.get_group_linear_id()] = local_mem[0];
                        }
                    });
                });

                len = n_wgroups;
            }


            sycl::buffer<T> recov {1};

            q.submit([&] (sycl::handler& cgh) {

                sycl::accessor global_mem {buf_int,cgh, sycl::read_only};
                sycl::accessor acc_rec {recov,cgh, sycl::write_only,sycl::no_init};
                
                cgh.single_task([=] () {
                    acc_rec[0] = global_mem[0];
                });
            });



            T rec;
            {
                sycl::host_accessor acc{recov,sycl::read_only};
                rec = acc[0];
            }

            return rec;



        }

        
    } // namespace reduction

    namespace convert {
        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> && vec);

        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> & vec);
    } // namespace convert

} // namespace syclalgs