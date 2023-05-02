// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/sycl.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamalgs::reduction {

    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);
    
    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    bool is_all_true(sycl::buffer<u8> & buf,u32 cnt);

    template<class T> bool equals(sycl::buffer<T> &buf1, sycl::buffer<T> &buf2, u32 cnt){

            sycl::buffer<u8> res (cnt);
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor acc1 {buf1, cgh, sycl::read_only};
                sycl::accessor acc2 {buf2, cgh, sycl::read_only};

                sycl::accessor out {res, cgh, sycl::write_only, sycl::no_init};
                
                cgh.parallel_for( sycl::range{cnt}, [=](sycl::item<1> item) { 
                    out[item] = shambase::vec_equals(acc1[item],acc2[item]); });
            });


            return shamalgs::reduction::is_all_true(res,cnt);

        }
} // namespace shamalgs::reduction
