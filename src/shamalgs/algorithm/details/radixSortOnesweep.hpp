// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shambase/integer.hpp"
#include "shambase/sycl.hpp"
#include "shambase/type_traits.hpp"
#include "shamsys/legacy/log.hpp"

#include <numeric>



namespace shamalgs::algorithm::details {

    


    template<class Tkey, class Tval>
    void sort_by_key_radix_onesweep_v1(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    ){

        sycl::buffer<Tkey> tmp_buf_key(len);
        sycl::buffer<Tval> tmp_buf_values(len);

        auto get_in_keys = [&](u32 step) -> sycl::buffer<Tkey>& {
            if(step%2 == 0){
                return buf_key;
            }else{
                return tmp_buf_key;
            }
        };

        auto get_out_keys = [&](u32 step) -> sycl::buffer<Tkey>& {
            if(step%2 == 0){
                return tmp_buf_key;
            }else{
                return buf_key;
            }
        };

        auto get_in_vals = [&](u32 step) -> sycl::buffer<Tval>& {
            if(step%2 == 0){
                return buf_values;
            }else{
                return tmp_buf_values;
            }
        };

        auto get_out_vals = [&](u32 step) -> sycl::buffer<Tval>& {
            if(step%2 == 0){
                return tmp_buf_values;
            }else{
                return buf_values;
            }
        };


        //shamalgs::memory::print_buf(get_in_keys(0), len, 16, "{:4} ");

        for(Tkey digit_offset = 0; digit_offset < shambase::bitsizeof<Tkey>; digit_offset ++){

            sycl::buffer<u32> digit_val(len);

            
            q.submit([&,len, digit_offset](sycl::handler &cgh) {
                sycl::accessor m{get_in_keys(digit_offset), cgh, sycl::read_only};
                sycl::accessor digit{digit_val, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                    u32 gid = item.get_linear_id();
                    digit[gid] = shambase::select_bit(m[gid], digit_offset);
                });
            });

            sycl::buffer<u32> offset_buf = numeric::exclusive_sum(q, digit_val, len);

            /*
            * num_zeros = len - offset_buf[len-1]  (+1 if digit_val[len-1] == 1)
            * if(digit == 0) offset = - offset_buf[gid]
            * else offset = num_zeros + offset_buf[gid]
            */

            u32 last_digit = memory::extract_element(q, digit_val, len-1);
            u32 last_offset = memory::extract_element(q, offset_buf, len-1);

            u32 one_offset = (len-1) - last_offset  + (last_digit == 1 ? 0 : 1);

            q.submit([&,len,one_offset](sycl::handler &cgh) {
                sycl::accessor keys{get_in_keys(digit_offset), cgh, sycl::read_only};
                sycl::accessor vals{get_in_vals(digit_offset), cgh, sycl::read_only};

                sycl::accessor digit{digit_val, cgh, sycl::read_only};
                sycl::accessor offset_scan{offset_buf, cgh, sycl::read_only};

                sycl::accessor new_keys{get_out_keys(digit_offset), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor new_vals{get_out_vals(digit_offset), cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                    u32 gid = item.get_linear_id();
                    
                    u32 cur_digit = digit[gid];
                    u32 doffset = offset_scan[gid];

                    u32 new_id = (cur_digit == 0) ? (gid-doffset) : (doffset + one_offset);

                    new_keys[new_id] = keys[gid];
                    new_vals[new_id] = vals[gid];

                });
            });

            //logger::raw_ln("step :", digit_offset);
            //shamalgs::memory::print_buf(get_out_keys(digit_offset), len, 16, "{:4} ");

        }



    }





    template<class Tkey, class Tval, u32 group_size>
    void sort_by_key_radix_onesweep_v2(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    ){

        sycl::buffer<Tkey> tmp_buf_key(len);
        sycl::buffer<Tval> tmp_buf_values(len);

        auto get_in_keys = [&](u32 step) -> sycl::buffer<Tkey>& {
            if(step%2 == 0){
                return buf_key;
            }else{
                return tmp_buf_key;
            }
        };

        auto get_out_keys = [&](u32 step) -> sycl::buffer<Tkey>& {
            if(step%2 == 0){
                return tmp_buf_key;
            }else{
                return buf_key;
            }
        };

        auto get_in_vals = [&](u32 step) -> sycl::buffer<Tval>& {
            if(step%2 == 0){
                return buf_values;
            }else{
                return tmp_buf_values;
            }
        };

        auto get_out_vals = [&](u32 step) -> sycl::buffer<Tval>& {
            if(step%2 == 0){
                return tmp_buf_values;
            }else{
                return buf_values;
            }
        };


        //shamalgs::memory::print_buf(get_in_keys(0), len, 16, "{:4} ");


        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt*group_size;

        sycl::buffer<u32> digit_val(corrected_len);
        sycl::buffer<u32> offset_buf(corrected_len);

        sycl::buffer<u64> tile_state (group_cnt);
        constexpr u32 STATE_X = 0;
        constexpr u32 STATE_A = 1;
        constexpr u32 STATE_P = 2;


        for(Tkey digit_offset = 0; digit_offset < shambase::bitsizeof<Tkey>; digit_offset ++){

            //shift by one for inclusive scan and compute digit
            q.submit([&,len, digit_offset](sycl::handler &cgh) {
                sycl::accessor m{get_in_keys(digit_offset), cgh, sycl::read_only};
                sycl::accessor digit{digit_val, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor offset{offset_buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                    u32 gid = item.get_linear_id();
                    {
                        u32 val = (gid < len) ? m[gid] : 0;
                        u32 digit_val = shambase::select_bit(val, digit_offset);
                        digit[gid] = digit_val;
                    }
                    {
                        u32 val = (gid > 0 && gid < len) ? m[gid-1] : 0;
                        u32 digit_val = shambase::select_bit(val, digit_offset);
                        offset[gid] = digit_val;
                    }
                });
            });



            shamalgs::memory::buf_fill_discard(q, tile_state, shambase::pack(STATE_X, u32(0)));


            q.submit([&, group_cnt, len](sycl::handler &cgh) {

                sycl::accessor acc_value {offset_buf, cgh, sycl::read_write};
                sycl::accessor acc_tile_state       {tile_state      , cgh, sycl::read_write};

                sycl::local_accessor<u32,1> local_scan_buf {1,cgh};
                sycl::local_accessor<u32,1> local_sum {1,cgh};

                using atomic_ref_T = sycl::atomic_ref<
                        u64, 
                        sycl::memory_order_relaxed, 
                        sycl::memory_scope_work_group,
                        sycl::access::address_space::global_space>;


                cgh.parallel_for(
                    sycl::nd_range<1>{corrected_len, group_size},
                    [=](sycl::nd_item<1> id) {

                        u32 local_id = id.get_local_id(0);
                        u32 group_tile_id = id.get_group_linear_id();
                        u32 global_id = group_tile_id * group_size + local_id;

                        //load from global buffer
                        u32 local_val = acc_value[global_id];
                        
                        //local scan in the group 
                        //the local sum will be in local id `group_size - 1`
                        u32 local_scan = sycl::inclusive_scan_over_group(id.get_group(), local_val, sycl::plus<u32>{});

                        if(local_id == group_size-1){
                            local_scan_buf[0] = local_scan;
                        }

                        //sync group
                        id.barrier(sycl::access::fence_space::local_space);

                        //DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                        if (local_id == 0) {

                            atomic_ref_T tile_atomic (acc_tile_state[group_tile_id]);

                            //load group sum
                            u32 local_group_sum = local_scan_buf[0];
                            u32 accum = 0;
                            u32 tile_ptr = group_tile_id-1;
                            sycl::vec<u32, 2> tile_state = {STATE_X,0};


                            //global scan using atomic counter

                            if (group_tile_id != 0)  {

                                tile_atomic.store(shambase::pack(STATE_A,local_group_sum));
                                
                                while (tile_state.x() != STATE_P){

                                    atomic_ref_T atomic_state (acc_tile_state[tile_ptr]);

                                    do{
                                        tile_state = shambase::unpack(atomic_state.load());
                                    }while(tile_state.x() == STATE_X);

                                    accum += tile_state.y();

                                    tile_ptr --;
                                }

                            }

                            tile_atomic.store(shambase::pack(STATE_P,accum + local_group_sum));

                            local_sum[0] = accum;
                        }

                        //sync
                        id.barrier(sycl::access::fence_space::local_space);

                        //store final result
                        //acc_value[global_id] = local_scan + local_sum[0] ;
                        acc_value[global_id] = local_scan + local_sum[0];
                        
                    }
                );
            });





            u32 last_digit = memory::extract_element(q, digit_val, len-1);
            u32 last_offset = memory::extract_element(q, offset_buf, len-1);

            u32 one_offset = (len-1) - last_offset  + (last_digit == 1 ? 0 : 1);

            q.submit([&,len,one_offset](sycl::handler &cgh) {
                sycl::accessor keys{get_in_keys(digit_offset), cgh, sycl::read_only};
                sycl::accessor vals{get_in_vals(digit_offset), cgh, sycl::read_only};

                sycl::accessor digit{digit_val, cgh, sycl::read_only};
                sycl::accessor offset_scan{offset_buf, cgh, sycl::read_only};

                sycl::accessor new_keys{get_out_keys(digit_offset), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor new_vals{get_out_vals(digit_offset), cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                    u32 gid = item.get_linear_id();
                    
                    u32 cur_digit = digit[gid];
                    u32 doffset = offset_scan[gid];

                    u32 new_id = (cur_digit == 0) ? (gid-doffset) : (doffset + one_offset);

                    new_keys[new_id] = keys[gid];
                    new_vals[new_id] = vals[gid];

                });
            });

            //logger::raw_ln("step :", digit_offset);
            //shamalgs::memory::print_buf(get_out_keys(digit_offset), len, 16, "{:4} ");

        }



    }




















    template<class T, u32 digit_bit_len>
    class DigitBinner{public:
        static constexpr T bitlen_T = shambase::bitsizeof<T>;
        static constexpr T digit_bit_places = bitlen_T / digit_bit_len;
        static constexpr T digit_count = (1U << digit_bit_len);
        static constexpr T value_count = digit_bit_places * digit_count;

        static_assert(digit_bit_places*digit_bit_len == bitlen_T, "the conversion should be correct");

        template<class Acc>
        static void fetch_add_bin(Acc accessor, T digit_val, T digit_place){
            using atomic_ref_T = sycl::atomic_ref<
                    u32, 
                    sycl::memory_order_relaxed, 
                    sycl::memory_scope_work_group,
                    sycl::access::address_space::local_space>;

            atomic_ref_T (accessor[digit_val + digit_place*digit_count]).fetch_add(1U);
        }

        template<class Acc>
        static void add_bin_key(Acc accessor, T value_to_bin){

            T digit_mask = digit_count-1;

            for(T digit_place = 0; digit_place < digit_bit_places; digit_place ++){
                T shifted = digit_mask & (value_to_bin >> (digit_place*digit_bit_len));

                fetch_add_bin(accessor, shifted, digit_place);
            }

        }

        template<u32 group_size, class Tkey>
        static sycl::buffer<u32> make_digit_histogram (sycl::queue & q, sycl::buffer<Tkey> &buf_key,u32 len){


            u32 group_cnt = shambase::group_count(len, group_size);

            group_cnt = group_cnt + (group_cnt % 4);
            u32 corrected_len = group_cnt*group_size;

            sycl::buffer<u32> digit_histogram(value_count);

            shamalgs::memory::buf_fill_discard(q, digit_histogram, 0U);

            logger::raw_ln("digit binning");
            memory::print_buf(digit_histogram, value_count, digit_count, "{:4} ");

            q.submit([&,len](sycl::handler &cgh) {
                sycl::accessor keys{buf_key, cgh, sycl::read_only};
                sycl::accessor histogram{digit_histogram, cgh, sycl::read_write};

                sycl::local_accessor<u32,1> local_histogram{value_count,cgh};

                cgh.parallel_for(sycl::nd_range<1>{corrected_len, group_size},
                    [=](sycl::nd_item<1> id) {
                    u32 local_id = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 global_id = group_tile_id * group_size + local_id;

                    //load from global buffer
                    if(global_id < len){
                        add_bin_key(local_histogram, keys[global_id]);
                    }

                    
                    for(u32 i = local_id ; i < value_count; i+= group_size){
                        u32 dcount = local_histogram[i];

                        if(dcount != 0){

                            using atomic_ref_t = sycl::atomic_ref<
                                u32, 
                                sycl::memory_order_relaxed, 
                                sycl::memory_scope_device,
                                sycl::access::address_space::global_space>;

                            atomic_ref_t(histogram[i]).fetch_add(dcount);
                        }
                    }
                });
            });

            return digit_histogram;
        }
    };

    /*
    tile histogram : 

    element a :

    a = 2 + 1x4^3

    |digit | digit places  |
    |      | 0 | 1 | 2 | 3 |
    ------------------------
    |  0   | 0 | 1 | 1 | 0 |
    |  1   | 0 | 0 | 0 | 1 |
    |  2   | 1 | 0 | 0 | 0 |
    |  3   | 0 | 0 | 0 | 0 |

    sum array on the table

    */
    template<class Tkey, class Tval, u32 group_size>
    void sort_by_key_radix_onesweep_v3(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    ){

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt*group_size;

        memory::print_buf(buf_key, len, 16, "{:4} ");

        constexpr u32 digit_len = 4;
        using Binner = DigitBinner<Tkey,digit_len>;

        sycl::buffer<u32> digit_histogram = Binner::template make_digit_histogram<group_size>(q, buf_key, len);

        logger::raw_ln("digit histogram");
        memory::print_buf(digit_histogram, Binner::value_count, Binner::digit_count, "{:4} ");


        {

            sycl::host_accessor acc {digit_histogram, sycl::read_write};

            for(u32 digit_place = 0; digit_place < Binner::digit_bit_places; digit_place++){
                u32 offset_ptr = Binner::digit_count*digit_place;
                std::exclusive_scan(
                    acc.get_pointer()+ offset_ptr , 
                    acc.get_pointer()+ offset_ptr + Binner::digit_count,
                    acc.get_pointer()+ offset_ptr ,0);
            }

        }


        logger::raw_ln("digit histogram");
        memory::print_buf(digit_histogram, Binner::value_count, Binner::digit_count, "{:4} ");

    }















}