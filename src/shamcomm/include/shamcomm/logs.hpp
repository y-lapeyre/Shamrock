// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file logs.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/string.hpp"
#include <string>

namespace shamcomm::logs {
    namespace details {
        inline i8 loglevel = 0;
    } // namespace details

    inline std::string format_message() { return ""; }

    template<typename T, typename... Types>
    std::string format_message(T var1, Types... var2);

    template<typename... Types>
    inline std::string format_message(std::string s, Types... var2) {
        return s + " " + format_message(var2...);
    }

    template<typename T, typename... Types>
    inline std::string format_message(T var1, Types... var2) {
        if constexpr (std::is_same_v<T, const char*>){
            return std::string(var1) + " " + format_message(var2...);
        }else if constexpr (std::is_pointer_v<T>){
            return shambase::format("{} ", static_cast<void *>(var1)) + format_message(var2...);
        }else {
            return shambase::format("{} ", var1) + format_message(var2...);
        }
    }

} // namespace shamcomm::logs