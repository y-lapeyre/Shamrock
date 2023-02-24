// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "log/fmt_defs.hpp"

namespace shamsys {

    template<typename... T>
    std::string format(fmt::format_string<T...> fmt, T &&...args) {
        return fmt::format(fmt, args...);
    }

} // namespace shamsys