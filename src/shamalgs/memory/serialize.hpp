// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/memory.hpp"
#include "shambase/sycl.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/NodeInstance.hpp"
#include <hipSYCL/sycl/buffer.hpp>
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <hipSYCL/sycl/libkernel/range.hpp>
#include <memory>
#include <type_traits>

namespace shamalgs::serialize {

    
    


    /**
     * @brief store a value of type T in a byte buffer
     *
     * @tparam T
     * @param buf
     * @param ptr_write
     * @param a
     */
    template<class T>
    inline void store(sycl::buffer<u8> &byte_buf, u64 ptr_write, T a) {
        shamsys::instance::get_compute_queue().submit([&, a, ptr_write](sycl::handler &cgh) {
            sycl::accessor accbuf{byte_buf, cgh, sycl::write_only};
            cgh.single_task([=]() { shambase::store_conv(&accbuf[ptr_write], a); });
        });
    }

    /**
     * @brief load a value of type T from a byte buffer
     *
     * @tparam T
     * @param buf
     * @param ptr_load
     * @return T
     */
    template<class T>
    inline T load(sycl::buffer<u8> &byte_buf, u64 ptr_load) {

        sycl::buffer<T> retbuf(1);

        shamsys::instance::get_compute_queue().submit([&, ptr_load](sycl::handler &cgh) {
            sycl::accessor accbuf{byte_buf, cgh, sycl::read_only};
            sycl::accessor retacc{retbuf, cgh, sycl::write_only, sycl::no_init};
            cgh.single_task([=]() { retacc[0] = shambase::load_conv<T>(&accbuf[ptr_load]); });
        });

        T ret_val;
        {
            sycl::host_accessor acc{retbuf, sycl::read_only};
            ret_val = acc[0];
        }

        return ret_val;
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param byte_buf 
     * @param ptr_write 
     * @param buf 
     * @param lenbuf 
     */
    template<class T>
    inline void store(sycl::buffer<u8> &byte_buf, u64 ptr_write, sycl::buffer<T> &buf, u32 lenbuf) {

        shamsys::instance::get_compute_queue().submit([&, ptr_write](sycl::handler &cgh) {
            sycl::accessor accbufbyte{byte_buf, cgh, sycl::write_only};
            sycl::accessor accbuf{buf, cgh, sycl::read_only};

            cgh.memcpy(
                accbufbyte.get_pointer() + ptr_write, accbuf.get_pointer, lenbuf * sizeof(T));
        });
    }

    template<class T, int n>
    inline void store(sycl::buffer<u8> &byte_buf, u64 ptr_write, sycl::buffer<sycl::vec<T, n>> &buf, u32 lenbuf) {

    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param byte_buf 
     * @param ptr_load 
     * @param lenbuf 
     * @return sycl::buffer<T> 
     */
    template<class T>
    inline sycl::buffer<T> load(sycl::buffer<u8> &byte_buf, u64 ptr_load, u32 lenbuf) {

        sycl::buffer<T> buf(lenbuf);

        shamsys::instance::get_compute_queue().submit([&, ptr_load](sycl::handler &cgh) {
            sycl::accessor accbufbyte{byte_buf, cgh, sycl::read_only};
            sycl::accessor accbuf{buf, cgh, sycl::write_only, sycl::no_init};

            cgh.memcpy(accbuf.get_pointer, accbufbyte.get_pointer() + ptr_load, lenbuf * sizeof(T));
        });
    }

} // namespace shamalgs::serialize