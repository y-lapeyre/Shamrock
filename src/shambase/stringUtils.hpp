// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "fmt_bindings/fmt_defs.hpp"
#include "exception.hpp"
#include <vector>

namespace shambase {

    template<typename... T>
    inline std::string format(fmt::format_string<T...> fmt, T &&...args) {
        try {
            return fmt::format(fmt, args...);
        } catch (const std::exception &e) {
            throw throw_with_loc<std::invalid_argument>("format failed : " + std::string(e.what()));
        }
    }

    template<typename... T>
    inline std::string format_printf(std::string format, const T & ...args) {
        try {
            return fmt::sprintf(format, args...);
        } catch (const std::exception &e) {

            throw throw_with_loc<std::invalid_argument>(
                "format failed : " + std::string(e.what()) +
                "\n fmt string : " + std::string(format)
            );
        }
    }

    inline std::string readable_sizeof(double size) {

        i32 i = 0;
        std::array units{"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};

        if (size >= 0) {
            while (size > 1024) {
                size /= 1024;
                i++;
            }
        } else {
            i = 9;
        }

        if (i > 8) {
            return format_printf("%s", "err val");
        } else {
            return format_printf("%.2f %s", size, units[i]);
        }
    }

} // namespace shambase