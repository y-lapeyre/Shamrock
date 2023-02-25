// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamutils/SourceLocation.hpp"

namespace shamutils{
    template<class ExcptTypes>
    inline ExcptTypes throw_with_loc(std::string message, SourceLocation loc = SourceLocation{}){
        return ExcptTypes(message + loc.format_multiline());
    }
}

