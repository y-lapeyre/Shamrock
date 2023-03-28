// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestAssert.hpp"

namespace shamtest::details {

    std::string TestAssert::serialize() {
        std::string acc = "\n{\n";

        acc += R"(    "value" : )" + std::to_string(value) + ",\n";
        acc += R"(    "name" : ")" + name + "\"";

        if (!comment.empty()) {
            acc += ",\n"
                   R"(    "comment" : ")" +
                   comment + "\"";
        }

        acc += "\n}";
        return acc;
    }

} // namespace shamtest::details