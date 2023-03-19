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
#include "shamrock/legacy/utils/time_utils.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/random/random.hpp"
#include <numeric>

template<class T>
struct TestExclScan {

    using vFunctionCall = sycl::buffer<T> (*)(sycl::queue &, sycl::buffer<T> &, u32);

    vFunctionCall fct;

    explicit TestExclScan(vFunctionCall arg) : fct(arg){};

    void check() {
        if constexpr (std::is_same<u32, T>::value) {

            u32 len_test = 1e5;

            std::vector<u32> data = shamalgs::random::mock_vector<u32>(0x111, len_test, 0, 10);

            std::vector<u32> data_buf(data);

            std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);

            sycl::buffer<u32> buf{data_buf.data(), data_buf.size()};

            //shamalgs::memory::print_buf(buf, 4096, 16, "{:4} ");

            sycl::buffer<u32> res =
                fct(shamsys::instance::get_compute_queue(), buf, data_buf.size());

            //shamalgs::memory::print_buf(res, len_test, 16, "{:4} ");
            
            bool eq = true;
            {
                sycl::host_accessor acc{res, sycl::read_only};

                for (u32 i = 0; i < data_buf.size(); i++) {
                    //shamtest::asserts().assert_equal("inclusive scan elem", acc[i], data[i]);
                    eq = eq && (acc[i] == data[i]);
                }
            }

            //sycl::buffer<u32> tmp (data.data(), data.size());
            //shamalgs::memory::print_buf(tmp, len_test, 16, "{:4} ");

            shamtest::asserts().assert_bool("exclusive scan match std", eq);
        }
    }

    f64 benchmark_one(u32 len){

        sycl::buffer<u32> buf = shamalgs::random::mock_buffer<u32>(0x111, len, 0, 50);

        sycl::queue & q = shamsys::instance::get_compute_queue();

        shamalgs::memory::move_buffer_on_queue(q, buf);

        q.wait();

        Timer t;
        t.start();
        sycl::buffer<u32> res = fct(q, buf, len);

        q.wait();t.end();

        return (t.nanosec*1e-9);

    }

    f64 bench_one_avg(u32 len){
        f64 sum = 0;

        f64 cnt = 4;

        if(len < 2e6){
            cnt = 10;
        }else if(len < 1e5){
            cnt = 100;
        }else if(len < 1e4){
            cnt = 1000;
        }


        for(u32 i = 0; i < cnt; i++){
            sum += benchmark_one(len);
        }

        return sum / cnt;
    }

    struct BenchRes{
        std::vector<f64> sizes;
        std::vector<f64> times;
    };

    BenchRes benchmark(u32 lim_bench = 1e8){
        BenchRes ret;
        
        logger::info_ln("TestExclScan","testing :",__PRETTY_FUNCTION__);

        for(f64 i = 1e3; i < lim_bench; i*=1.1){
            ret.sizes.push_back(i);
        }

        for(const f64 & sz : ret.sizes){
            logger::debug_ln("ShamrockTest","N=",sz);
            ret.times.push_back(bench_one_avg(sz));
        }

        return ret;
    }

};


struct TestStreamCompact {

    using vFunctionCall = std::tuple<std::optional<sycl::buffer<u32>>, u32> (*)(sycl::queue&, sycl::buffer<u32> &, u32 );

    vFunctionCall fct;

    explicit TestStreamCompact(vFunctionCall arg) : fct(arg){};

    void check() {
        std::vector<u32> data {1,0,0,1,0,1,1,0,1,0,1,0,1};

        u32 len = data.size();

        auto buf = shamalgs::memory::vec_to_buf(data);

        auto [res, res_len] = fct(shamsys::instance::get_compute_queue(), buf, len);

        auto res_check = shamalgs::memory::buf_to_vec(*res, res_len);

        //make check
        std::vector<u32> idxs ;
        {
            for(u32 idx = 0; idx < len; idx ++){
                if(data[idx]){
                    idxs.push_back(idx);
                }
            }
        }

        shamtest::asserts().assert_equal("same lenght", res_len, u32(idxs.size()));

        for(u32 idx = 0; idx < res_len; idx ++){
            shamtest::asserts().assert_equal("sid_check", res_check[idx], idxs[idx]);
        }
    }
};