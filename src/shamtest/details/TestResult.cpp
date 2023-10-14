// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestResult.hpp"


#include "shamsys/NodeInstance.hpp"
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

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
                   R"(    "asserts" : )" + shambase::increase_indent(asserts.serialize_json()) + ",\n" +
                   R"(    "test_data" : )" + shambase::increase_indent(test_data.serialize_json()) + "\n" + "}";
        };

        return get_str();
    }



    std::basic_string<u8> TestResult::serialize() {

        std::basic_stringstream<u8> out;

        out.write(reinterpret_cast<u8 const*>(&type), sizeof(type));

        u64 name_len = name.size();
        out.write(reinterpret_cast<u8 const*>(&name_len), sizeof(name_len));
        out.write(reinterpret_cast<u8 const*>(name.data()), name_len * sizeof(char));

        out.write(reinterpret_cast<u8 const*>(&world_rank), sizeof(world_rank));

        std::basic_string<u8> res_assert = asserts.serialize();
        std::basic_string<u8> res_test_data = test_data.serialize();

        u64 res_assert_len = res_assert.size();
        out.write(reinterpret_cast<u8 const*>(&res_assert_len), sizeof(res_assert_len));
        out.write(reinterpret_cast<u8 const*>(res_assert.data()), res_assert_len * sizeof(char));

        u64 res_test_data_len = res_test_data.size();
        out.write(reinterpret_cast<u8 const*>(&res_test_data_len), sizeof(res_test_data_len));
        out.write(reinterpret_cast<u8 const*>(res_test_data.data()), res_test_data_len * sizeof(char));


        u64 tex_out_len = name.size();
        out.write(reinterpret_cast<u8 const*>(&tex_out_len), sizeof(tex_out_len));
        out.write(reinterpret_cast<u8 const*>(tex_output.data()), tex_out_len * sizeof(char));

        return out.str();
    }

} // namespace shamtest::details