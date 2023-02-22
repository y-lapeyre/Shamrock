// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/memory/memory.hpp"
#include "shamalgs/numeric/details/numericFallback.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>


#include "numeric_tests.hpp"


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_fallback", exclsumtest_fallback, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_fallback);
    test.check();
}



TestStart(Unittest, "shamalgs/numeric/details/stream_compact_fallback", streamcompactalg_fallback, 1){
    TestStreamCompact test ((TestStreamCompact::vFunctionCall)shamalgs::numeric::details::stream_compact_fallback);
    test.check();

}