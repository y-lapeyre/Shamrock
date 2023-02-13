// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"


namespace shamalgs::algorithm {



    //template<class T>
    //void sort(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    /**
     * @brief Sort the buffer according to the key order
     * 
     * @tparam T 
     * @param q 
     * @param buf_key 
     * @param buf_values 
     * @param len 
     */
    template<class Tkey, class Tval>
    void sort_by_key(sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);


    

    template<class Fct>
    inline sycl::buffer<typename std::invoke_result_t<Fct,u32>> gen_buffer_device(sycl::queue & q, u32 len, Fct && func){

        using ret_t = typename std::invoke_result_t<Fct,u32>;

        sycl::buffer<ret_t> ret(len);

        q.submit([&](sycl::handler &cgh) {

            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>(len),
                [=](sycl::item<1> item) {
                    out[item] = func(item.get_linear_id());
                }
            );
        });

        return std::move(ret);
    }

    sycl::buffer<u32> gen_buffer_index(sycl::queue & q , u32 len);

}