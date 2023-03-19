// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/atomic/DeviceCounter.hpp"
#include "shamalgs/atomic/DynamicIdGenerator.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shambase/integer.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T, u32 group_size>
    inline void exclusive_sum_atomic_decoupled_v6_in_place(sycl::queue &q, sycl::buffer<T> &buf, u32 len){

    }


    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v6;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic_decoupled_v6(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt*group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(corrected_len);

        //logger::raw_ln("shifted : ");
        //shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<u64> tile_state (group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;


        shamalgs::memory::buf_fill_discard(q, tile_state, shambase::pack(STATE_X, T(0)));

        q.submit([&, group_cnt, len](sycl::handler &cgh) {

            sycl::accessor acc_in {buf1, cgh, sycl::read_only};
            sycl::accessor acc_out {ret_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_tile_state       {tile_state      , cgh, sycl::read_write};

            sycl::local_accessor<T,1> local_scan_buf {1,cgh};
            sycl::local_accessor<T,1> local_sum {1,cgh};


            using atomic_ref_T = sycl::atomic_ref<
                    u64, 
                    sycl::memory_order_relaxed, 
                    sycl::memory_scope_work_group,
                    sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v6<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size},
                [=](sycl::nd_item<1> id) {

                    u32 local_id = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 global_id = group_tile_id * group_size + local_id;

                    auto local_group = id.get_group();

                    //load from global buffer
                    T local_val = (global_id > 0 && global_id < len) ? acc_in[global_id - 1] : 0;;
                    
                    //local scan in the group 
                    //the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(local_group, local_val, sycl::plus<>());

                    if(local_id == group_size-1){
                        local_scan_buf[0] = local_scan;
                    }

                    //sync group
                    id.barrier(sycl::access::fence_space::local_space);


                    //parralelized lookback
                    constexpr u32 thread_counts = 8;
                    static_assert(thread_counts <= group_size, "impossible");

                    
                        
                    T local_group_sum = local_scan_buf[0];
                    T accum = 0;

                    if (group_tile_id != 0)  {
                        if (local_id == 0) {
                            atomic_ref_T(acc_tile_state[group_tile_id]).store(shambase::pack(STATE_A,local_group_sum));
                        }

                        sycl::vec<T, 2> tile_state;
                        u32 group_tile_ptr = group_tile_id-1;
                        
                        bool continue_loop = true;

                        do{

                            if((local_id < thread_counts) && (group_tile_ptr >= local_id)){
                                atomic_ref_T atomic_state (acc_tile_state[group_tile_ptr - local_id]);

                                do{
                                    tile_state = shambase::unpack(atomic_state.load());
                                }while(tile_state.x() == STATE_X);

                            }else{
                                tile_state = {STATE_A,0};
                            }

                            T sum_state = sycl::reduce_over_group(local_group, tile_state.x(), sycl::plus<>());

                            if(sum_state > group_size){
                                //there is a P

                                continue_loop = false;

                                u32 last_p_index = sycl::reduce_over_group(local_group, (tile_state.x() == STATE_P) ? (local_id) : (0), sycl::maximum<>());

                                tile_state.y() = (local_id >= last_p_index) ? tile_state.y() : 0;

                            }else{
                                //there is only A's
                                continue_loop = (group_tile_ptr >= thread_counts);
                                group_tile_ptr -= thread_counts;
                            }

                            accum += sycl::reduce_over_group(local_group, tile_state.y(), sycl::plus<>());

                        }while(continue_loop);
                    }

                    if (local_id == 0) {
                        atomic_ref_T(acc_tile_state[group_tile_id]).store(shambase::pack(STATE_P,accum + local_group_sum));
                    }
                    
                    //store final result
                    if(global_id < len){
                        acc_out[global_id] = accum + local_scan;
                    }
                    
                }
            );
        });

        return ret_buf;

    }

}