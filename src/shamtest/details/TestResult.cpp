// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestResult.hpp"


#include "shamsys/NodeInstance.hpp"
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
                   R"(    "asserts" : )" + shambase::increase_indent(asserts.serialize()) + ",\n" +
                   R"(    "test_data" : )" + shambase::increase_indent(test_data.serialize()) + "\n" + "}";
        };

        return get_str();
    }


template<typename POD>
std::ostream& serialize(std::ostream& os, std::vector<POD> const& v)
{
    // this only works on built in data types (PODs)
    static_assert(std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
        "Can only serialize POD types with this function");

    auto size = v.size();
    os.write(reinterpret_cast<char const*>(&size), sizeof(size));
    os.write(reinterpret_cast<char const*>(v.data()), v.size() * sizeof(POD));
    return os;
}

template<typename POD>
std::istream& deserialize(std::istream& is, std::vector<POD>& v)
{
    static_assert(std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
        "Can only deserialize POD types with this function");

    decltype(v.size()) size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    v.resize(size);
    is.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(POD));
    return is;
}

    std::vector<u8> TestResult::serialize() {

        std::vector<u8> res;

        //TODO

        return res;
    }

} // namespace shamtest::details