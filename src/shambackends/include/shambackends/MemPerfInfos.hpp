// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MemPerfInfos.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include <cstddef>

namespace sham {

    struct MemPerfInfos {
        f64 time_alloc_host   = 0;
        f64 time_alloc_device = 0;
        f64 time_alloc_shared = 0;

        f64 time_free_host   = 0;
        f64 time_free_device = 0;
        f64 time_free_shared = 0;

        size_t allocated_byte_host   = 0;
        size_t allocated_byte_device = 0;
        size_t allocated_byte_shared = 0;
    };

} // namespace sham
