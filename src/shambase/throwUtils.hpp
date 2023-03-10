// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/SourceLocation.hpp"

namespace shambase {

    /**
     * @brief Throw an exception and append the source location to it
     *
     * Usage : 
     * ~~~~~{.cpp}
     * throw shambase::throw_with_loc<MyException>("message");
     * ~~~~~
     * 
     * @tparam ExcptTypes 
     * @param message 
     * @param loc 
     * @return ExcptTypes 
     */
    template<class ExcptTypes>
    inline ExcptTypes throw_with_loc(std::string message, SourceLocation loc = SourceLocation{}) {
        return ExcptTypes(message + loc.format_multiline());
    }

} // namespace shambase
