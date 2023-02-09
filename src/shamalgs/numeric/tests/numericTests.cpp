// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"


#include "shamalgs/numeric/numeric.hpp"
#include <numeric>

TestStart(Unittest, "shamalgs/numeric/exclusive_sum", exclsumtest, 1){
    
    std::vector<u32> data {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<u32> data_buf {3, 1, 4, 1, 5, 9, 2, 6};

    std::exclusive_scan(data.begin(), data.end(),data.begin(), 0);

    sycl::buffer<u32> buf {data_buf.data(), data_buf.size()};

    sycl::buffer<u32> res = shamalgs::numeric::exclusive_sum(shamsys::instance::get_compute_queue(), buf, data_buf.size());

    {
        sycl::host_accessor acc {res, sycl::read_only};

        for(u32 i = 0; i < data_buf.size(); i++){
            shamtest::asserts().assert_equal("inclusive scan elem", acc[i], data[i]);
        }
    }
}