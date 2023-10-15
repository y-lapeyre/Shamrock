// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "DataNode.hpp"
#include "shambase/string.hpp"

namespace shamtest::details {
    
    std::string serialize_vec(const std::vector<f64> &vec) {
        std::string acc = "\n[\n";

        for (u32 i = 0; i < vec.size(); i++) {
            acc += shambase::format_printf("%e", vec[i]);
            if (i < vec.size() - 1) {
                acc += ", ";
            }
        }

        acc += "\n]";
        return acc;
    }

    std::string DataNode::serialize_json() {
        std::string acc = "\n{\n";

        acc += R"(    "name" : ")" + name + "\",\n";

        acc += R"(    "data" : )"
               "\n" +
               serialize_vec(data) + "\n";

        acc += "\n}";
        return acc;
    }
} // namespace shamtest::details