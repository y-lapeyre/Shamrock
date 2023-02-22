// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sort_tests.hpp"
#include "shamalgs/algorithm/details/bitonicSort.hpp"

TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_legacy", test_bitonic_sort_legacy, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_legacy
        );
    test.check();
}


TestStart(Benchmark, "shamalgs/algorithm/details/bitonicSort_legacy:benchmark", 
    test_bitonic_sort_legacy_benchmark, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_legacy
        );
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}

TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_updated", test_bitonic_sort_updated, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,16>
        );
    test.check();
}


TestStart(Benchmark, "shamalgs/algorithm/details/bitonicSort_updated:benchmark", 
    test_bitonic_sort_updated_benchmark, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,16>
        );
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}