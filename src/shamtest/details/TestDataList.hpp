// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "TestData.hpp"
#include <vector>

namespace shamtest::details {

    struct TestDataList {
        std::vector<TestData> test_data;

        // define member function here
        // to register test data

        [[nodiscard]] inline TestData &new_dataset(std::string name) {
            test_data.push_back(TestData{std::move(name), {}});
            return test_data.back();
        }

        std::string serialize_json();
        std::basic_string<u8> serialize();
    };

}