// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "sycl_utils/sycl_utilities.hpp"
#include "sycl_utils/vectorProperties.hpp"
#include "sycl_utils/vec_equals.hpp"
#include <stdexcept>

namespace shambase {

    template<class T>
    void check_buffer_size(sycl::buffer<T> & buf, u64 max_range,
        struct SourceLocation loc = SourceLocation()){
        if(buf.size() < max_range){
            throw throw_with_loc<std::invalid_argument>("buffer is too small",loc);
        }
    }
}