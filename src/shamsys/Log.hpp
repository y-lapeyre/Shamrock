// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/stringUtils.hpp"

namespace shamsys {

    template<typename... T>
    inline std::string format(fmt::format_string<T...> fmt, T &&...args) {
        return shambase::format(fmt, args...);
    }

    template<typename... T>
    inline std::string format_printf(std::string format, T &&...args){
        return shambase::format_printf(format, args...);
    }



} // namespace shamsys