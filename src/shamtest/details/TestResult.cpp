// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestResult.hpp"


#include "shamsys/NodeInstance.hpp"

namespace shamtest::details {

    std::string TestResult::serialize_json() {

        using namespace shamsys::instance;

        auto get_type_name = [](TestType t) -> std::string {
            switch (t) {
            case Benchmark:
                return "Benchmark";
            case Analysis:
                return "Analysis";
            case Unittest:
                return "Unittest";
            }
        };

        auto get_str = [&]() -> std::string {
            return "{\n"
                   R"(    "type" : ")" +
                   get_type_name(type) + "\",\n" + R"(    "name" : ")" + name + "\",\n" +
                   R"(    "compute_queue" : ")" +
                   get_compute_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                   R"(    "alt_queue" : ")" +
                   get_alt_queue().get_device().get_info<sycl::info::device::name>() + "\",\n" +
                   R"(    "world_rank" : )" + std::to_string(world_rank) + ",\n" +
                   R"(    "asserts" : )" + shambase::increase_indent(asserts.serialize()) + ",\n" +
                   R"(    "test_data" : )" + shambase::increase_indent(test_data.serialize()) + "\n" + "}";
        };

        return get_str();
    }

} // namespace shamtest::details