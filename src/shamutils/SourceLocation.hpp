// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

struct SourceLocation {

    const char *fileName;
    const char *functionName;
    const u32 lineNumber;
    const u32 columnOffset;

    static constexpr SourceLocation current(

#if defined __has_builtin
    #if __has_builtin(__builtin_FILE)
        const char *fileName = __builtin_FILE(),
    #else
        const char *fileName     = "unimplemented",
    #endif
    #if __has_builtin(__builtin_FUNCTION)
        const char *functionName = __builtin_FUNCTION(),
    #else
        const char *functionName = "unimplemented",
    #endif
    #if __has_builtin(__builtin_LINE)
        const u32 lineNumber = __builtin_LINE(),
    #else
        const u32 lineNumber     = 0,
    #endif
    #if __has_builtin(__builtin_COLUMN)
        const u32 columnOffset = __builtin_COLUMN()
    #else
        const u32 columnOffset   = 0
    #endif
#else

        const char *fileName     = "unimplemented",
        const char *functionName = "unimplemented",
        const u32 lineNumber     = 0,
        const u32 columnOffset   = 0
#endif
    )

        noexcept {
        return SourceLocation{fileName, functionName, lineNumber, columnOffset};
    }
};