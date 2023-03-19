// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//



#include "numericTests.hpp"
#include "shamalgs/numeric/details/exclusiveScanAtomic.hpp"
#include "shamalgs/numeric/details/exclusiveScanGPUGems39.hpp"
#include "shamalgs/numeric/details/numericFallback.hpp"
#include "shamalgs/numeric/details/scanDecoupledLookback.hpp"
#include "shamalgs/numeric/numeric.hpp"


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_gpugems39", test_exclusive_sum_gpugems39_1, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_1);
    test.check();
}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic2pass", test_exclusive_sum_atomic2pass, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic2pass<u32,16>);
    test.check();

    //std::vector<u32> buf_init = {
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,
    //    1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1
    //};
    //
    //sycl::buffer<u32> buf(buf_init.data(), buf_init.size());
    //
    //shamalgs::memory::print_buf(buf, buf_init.size(), 16,"{:2} ");
    //
    //sycl::buffer<u32> ret = 
    //    shamalgs::numeric::details::exclusive_sum_atomic2pass<u32, 16>(
    //        shamsys::instance::get_compute_queue(), 
    //        buf, 
    //        buf_init.size());
    //
    //logger::raw_ln("scanned : ");
    //
    //shamalgs::memory::print_buf(ret, buf_init.size(), 16,"{:2} ");
}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic2pass_v2", test_exclusive_sum_atomic2pass_v2, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic2pass_v2<u32,16>);
    test.check();

}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled", test_exclusive_sum_atomic_decoupled, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v2", test_exclusive_sum_atomic_decoupled_v2, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,16>);
    test.check();

}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v3", test_exclusive_sum_atomic_decoupled_v3, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v4", test_exclusive_sum_atomic_decoupled_v4, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v4<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v5", test_exclusive_sum_atomic_decoupled_v5, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v6", test_exclusive_sum_atomic_decoupled_v6, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_sycl_jointalg", test_exclusive_sum_sycl_jointalg, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_sycl_jointalg<u32,32>);
    test.check();

}



TestStart(Benchmark, "shamalgs/numeric/details/exclusive_sum:benchmark", bench_exclusive_sum, 1){
/*
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
*/
    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_fallback);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("fallback u32");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

/* 
    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,256>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize = 256");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,1024>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize = 1024");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,64>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize = 64");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,128>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize = 128");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,256>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize = 256");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    } */

/*
    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,256>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize = 256");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,1024>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize = 1024");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }


    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v4<u32,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v4 u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

*/
    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5<u32,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v5 u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    //{
    //    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32,512>);
    //    auto result = test.benchmark();
//
    //    auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v6 u32 gsize = 512");
//
    //    res.add_data("Nobj", result.sizes);
    //    res.add_data("t_sort", result.times);
    //}

    {
        TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_sycl_jointalg<u32,512>);
        auto result = test.benchmark(1e7);

        auto & res = shamtest::test_data().new_dataset("sycl joint excl sum u32 gsize = 512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

}
