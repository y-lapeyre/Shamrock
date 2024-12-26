// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file groupReduction_usm.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/reduction/groupReduction_usm_impl.hpp"
#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    T sum_usm_group(
        sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size) {

        return reduc_internal<T>(
            sched,
            buf1,
            start_id,
            end_id,
            work_group_size,
            [](sycl::group<1> g, T v) {
                return sycl::reduce_over_group(g, v, SYCL_SUM_OP);
            },
            [](T lhs, T rhs) {
                return lhs + rhs;
            });
    }

} // namespace shamalgs::reduction::details

#define XMAC_TYPES                                                                                 \
    X(f32)                                                                                         \
    X(f32_2)                                                                                       \
    X(f32_3)                                                                                       \
    X(f32_4)                                                                                       \
    X(f32_8)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64)                                                                                         \
    X(f64_2)                                                                                       \
    X(f64_3)                                                                                       \
    X(f64_4)                                                                                       \
    X(f64_8)                                                                                       \
    X(f64_16)                                                                                      \
    X(u32)                                                                                         \
    X(u64)                                                                                         \
    X(i32)                                                                                         \
    X(i64)                                                                                         \
    X(u32_3)                                                                                       \
    X(u64_3)                                                                                       \
    X(i64_3)                                                                                       \
    X(i32_3)

#define X(_arg_)                                                                                   \
    template _arg_ shamalgs::reduction::details::sum_usm_group<_arg_>(                             \
        sham::DeviceScheduler_ptr & sched,                                                         \
        sham::DeviceBuffer<_arg_> & buf1,                                                          \
        u32 start_id,                                                                              \
        u32 end_id,                                                                                \
        u32 work_group_size);

XMAC_TYPES
#undef X
