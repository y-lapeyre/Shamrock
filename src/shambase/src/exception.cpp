// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file exception.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include <string>

namespace shambase {

    std::string exception_format(SourceLocation loc) {
        return loc.format_multiline(fmt_callstack());
    }

} // namespace shambase
