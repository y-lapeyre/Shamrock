// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "algorithm.hpp"
#include "details/bitonicSort.hpp"
namespace shamalgs::algorithm {

    template<class Tkey, class Tval>
    void sort_by_key(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    ) {
        details::sort_by_key_bitonic_updated<Tkey, Tval, 16>(q, buf_key, buf_values, len);
    }

    template void
    sort_by_key(sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void
    sort_by_key(sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    sycl::buffer<u32> gen_buffer_index(sycl::queue &q, u32 len) {
        return gen_buffer_device(q, len, [](u32 i) -> u32 { return i; });
    }
} // namespace shamalgs::algorithm