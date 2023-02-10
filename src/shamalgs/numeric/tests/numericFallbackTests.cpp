// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/memory/memory.hpp"
#include "shamalgs/numeric/details/fallbackNumeric.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"


TestStart(Unittest, "shamalgs/numeric/details/stream_compact_fallback", streamcompactalg_fallback, 1){
    
    std::vector<u32> data {1,0,0,1,0,1,1,0,1,0,1,0,1};

    u32 len = data.size();

    auto buf = shamalgs::memory::vec_to_buf(data);

    auto [res, res_len] = shamalgs::numeric::details::stream_compact_fallback(shamsys::instance::get_compute_queue(), buf, len);

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