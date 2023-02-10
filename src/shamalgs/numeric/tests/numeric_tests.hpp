// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>

template<class T>
struct TestExclScan {

    using vFunctionCall = sycl::buffer<T> (*)(sycl::queue &, sycl::buffer<T> &, u32);

    vFunctionCall fct;

    explicit TestExclScan(vFunctionCall arg) : fct(arg){};

    void check() {
        if constexpr (std::is_same<u32, T>::value) {
            std::vector<u32> data{3, 1, 4, 1, 5, 9, 2, 6};
            std::vector<u32> data_buf{3, 1, 4, 1, 5, 9, 2, 6};

            std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);

            sycl::buffer<u32> buf{data_buf.data(), data_buf.size()};

            sycl::buffer<u32> res =
                fct(shamsys::instance::get_compute_queue(), buf, data_buf.size());

            {
                sycl::host_accessor acc{res, sycl::read_only};

                for (u32 i = 0; i < data_buf.size(); i++) {
                    shamtest::asserts().assert_equal("inclusive scan elem", acc[i], data[i]);
                }
            }
        }
    }
};


struct TestStreamCompact {

    using vFunctionCall = std::tuple<sycl::buffer<u32>, u32> (*)(sycl::queue&, sycl::buffer<u32> &, u32 );

    vFunctionCall fct;

    explicit TestStreamCompact(vFunctionCall arg) : fct(arg){};

    void check() {
        std::vector<u32> data {1,0,0,1,0,1,1,0,1,0,1,0,1};

        u32 len = data.size();

        auto buf = shamalgs::memory::vec_to_buf(data);

        auto [res, res_len] = fct(shamsys::instance::get_compute_queue(), buf, len);

        auto res_check = shamalgs::memory::buf_to_vec(res, res_len);

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