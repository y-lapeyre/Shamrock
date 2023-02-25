// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris
// <timothee.david--cleris@ens-lyon.fr> Licensed under CeCILL 2.1 License, see
// LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamutils/SourceLocation.hpp"

template<class ExcptTypes, class ... T>
void throw_with_loc(std::string message, SourceLocation loc = SourceLocation{}){
    throw ExcptTypes(message + loc.format_multiline());
}