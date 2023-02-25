// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamsys/Log.hpp"

/**
 * @brief provide information abount the source location
 *
 */
struct SourceLocation {

    const char *fileName;
    const char *functionName;
    const u32 lineNumber;
    const u32 columnOffset;

    explicit SourceLocation(

        #if defined __has_builtin
        #if __has_builtin(__builtin_FILE)
            const char *fileName = __builtin_FILE(),
        #else
            const char *fileName = "unimplemented",
        #endif
        #if __has_builtin(__builtin_FUNCTION)
            const char *functionName = __builtin_FUNCTION(),
        #else
            const char *functionName = "unimplemented",
        #endif
        #if __has_builtin(__builtin_LINE)
            const u32 lineNumber = __builtin_LINE(),
        #else
            const u32 lineNumber = 0,
        #endif
        #if __has_builtin(__builtin_COLUMN)
            const u32 columnOffset = __builtin_COLUMN()
        #else
            const u32 columnOffset = 0
        #endif
        #else

            const char *fileName = "unimplemented",
            const char *functionName = "unimplemented",
            const u32 lineNumber = 0, const u32 columnOffset = 0
        #endif

        ) : 
            fileName(fileName), 
            functionName(functionName), 
            lineNumber(lineNumber),
            columnOffset(columnOffset) {}

    std::string format_multiline(){
        return shamsys::format(
R"=(
---- Source Location ----
{} {}:{}
call = {}
-------------------------
)="
            , fileName, lineNumber, columnOffset, functionName);
    }

    std::string format_one_line(){
        return shamsys::format("{}:{}:{}", fileName, lineNumber, columnOffset);
    }
};