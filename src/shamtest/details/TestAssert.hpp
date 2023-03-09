// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamutils/SourceLocation.hpp"

namespace shamtest::details {

    struct TestAssert {
        bool value;
        std::string name;
        std::string comment;

        std::string serialize();

    };

} // namespace shamtest::details