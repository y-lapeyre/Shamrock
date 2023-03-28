// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamtest/details/TestDataList.hpp"
#include "shamtest/details/TestAssertList.hpp"

enum TestType { Benchmark, Analysis, Unittest };

namespace shamtest::details {
    struct TestResult {
        TestType type;
        std::string name;
        u32 world_rank;
        TestAssertList asserts;
        TestDataList test_data;

        inline TestResult(const TestType &type, std::string name, const u32 &world_rank)
            : type(type), name(std::move(name)), world_rank(world_rank), asserts{}, test_data() {}

        std::string serialize();
    };
} // namespace shamtest::details