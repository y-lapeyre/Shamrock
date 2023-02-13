// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sort_tests.hpp"
#include "shamalgs/algorithm/algorithm.hpp"

TestStart(Unittest, "shamalgs/algorithm/sort_by_key", test_sort_by_key_func, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );
    test.check();
}


TestStart(Benchmark, "shamalgs/algorithm/sort_by_key:benchmark", 
    test_sort_by_key_func_benchmark, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}