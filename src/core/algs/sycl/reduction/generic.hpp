// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "core/algs/sycl/basic/basic.hpp"
#include <functional>

namespace syclalgs::reduction::impl {

    template<u32 work_group_size>
    struct manual_reduce_impl{
        template<class T, class Op> inline static T reduce_manual(sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id, Op op){

            u32 len = end_id - start_id;


            sycl::buffer<T> buf_int(len);

            #ifdef SYCL_COMP_HIPSYCL
            q.wait();
            #endif
            
            ::syclalgs::basic::write_with_offset_into(buf_int,buf1,start_id,len);

            #ifdef SYCL_COMP_HIPSYCL
            q.wait();
            #endif

            u32 part_size = work_group_size * 2;

            while (len != 1) {
                u32 n_wgroups = (len + part_size - 1) / part_size;    

                auto Bop = op;

                q.submit([&] (sycl::handler& cgh) {
                    sycl::local_accessor<T> local_mem {sycl::range<1>(work_group_size), cgh};

                    sycl::accessor global_mem {buf_int,cgh, sycl::read_write};
                    
                    cgh.parallel_for(
                        sycl::nd_range<1>(n_wgroups * work_group_size, work_group_size),
                        [=] (sycl::nd_item<1> item) {

                        size_t local_id = item.get_local_id(0);
                        size_t global_id = item.get_global_id(0);
                        local_mem[local_id] = 0;

                        if ((2 * global_id) < len) {
                            local_mem[local_id] = Bop(global_mem[2 * global_id] , global_mem[2 * global_id + 1]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        for (size_t stride = 1; stride < work_group_size; stride *= 2) {
                            auto idx = 2 * stride * local_id;
                            if (idx < work_group_size) {
                                local_mem[idx] = Bop(local_mem[idx] , local_mem[idx + stride]);
                            }

                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        if (local_id == 0) {
                            global_mem[item.get_group_linear_id()] = local_mem[0];
                        }
                    });
                });

                #ifdef SYCL_COMP_HIPSYCL
                q.wait();
                #endif

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

            #ifdef SYCL_COMP_HIPSYCL
            q.wait();
            #endif



            T rec;
            {
                sycl::host_accessor acc{recov,sycl::read_only};
                rec = acc[0];
            }

            return rec;



        }
    };
    


    template<class T, class Op> inline T reduce_sycl_2020(sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id, Op op){


        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);
        ::syclalgs::basic::write_with_offset_into(buf_int,buf1,start_id,len);

        sycl::buffer<T> recov {1};

        q.submit([&] (sycl::handler& cgh) {

            sycl::accessor global_mem {buf_int,cgh, sycl::read_only};

            #ifdef SYCL_COMP_DPCPP
            auto reduc = sycl::reduction(recov,cgh, op);
            #else
            sycl::accessor acc_rec {recov,cgh, sycl::write_only,sycl::no_init};
            auto reduc = sycl::reduction(acc_rec, op);
            #endif

            cgh.parallel_for(sycl::range<1>{len},reduc,
                [=](sycl::id<1> idx, auto& sum){
                    sum.combine(global_mem[idx]) ;
                }
            );
        });



        T rec;
        {
            sycl::host_accessor acc{recov,sycl::read_only};
            rec = acc[0];
        }

        return rec;

    }

}




namespace syclalgs::reduction {

    template<class T, class Op> inline T reduce(sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id, Op op){

        //return impl::reduce_sycl_2020<T, Op>(q, buf1, start_id, end_id,op);
        //return impl::reduce_manual<T, Op, 16>(q, buf1, start_id, end_id,op);
        return T{};
    }

}