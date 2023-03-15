// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//



#include "numericTests.hpp"
#include "shamalgs/numeric/details/exclusiveScanGPUGems39.hpp"
#include "shamalgs/numeric/details/numericFallback.hpp"
#include "shamalgs/numeric/numeric.hpp"


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_gpugems39", test_exclusive_sum_gpugems39_1, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_1);
    test.check();
}



TestStart(Benchmark, "shamalgs/numeric/details/exclusive_sum:benchmark", bench_exclusive_sum, 1){

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::exclusive_sum);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("public u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_1);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("gpugems39 v1 u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_2);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("gpugems39 v2 u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_3);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("gpugems39 v3 u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_fallback);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("fallback u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
}
