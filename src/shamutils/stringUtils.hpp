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
#include <vector>

namespace shamutils {

    template<typename... T>
    inline std::string format(fmt::format_string<T...> fmt, T &&...args) {
        return fmt::format(fmt, args...);
    }

    template<typename... T>
    inline std::string format_printf(std::string format, T &&...args){
        return fmt::sprintf(format, args...);
    }

    inline std::string readable_sizeof(double size){

        i32 i = 0;
        std::array units{"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};

        if(size >= 0){
            while (size > 1024) {
                size /= 1024;
                i++;
            }
        }else{
            i = 9;
        }

        if(i > 8){
            return format_printf("%s", "err val");
        }else{
            return format_printf( "%.2f %s", size, units[i]);
        }

    }



} // namespace shamsys