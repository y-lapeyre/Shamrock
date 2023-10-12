// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TestResult.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header describing return type of a test, and the type of the test
 * @date 2023-01-04
 *
 */

#include "aliases.hpp"
#include "shamtest/details/TestAssertList.hpp"
#include "shamtest/details/TestDataList.hpp"
#include <vector>

/**
 * @brief Describe the type of the performed test
 */
enum TestType { 
    Benchmark, 
    Analysis, 
    Unittest 
};

namespace shamtest::details {

    /**
     * @brief Result of a test
     */
    struct TestResult {
        TestType type;          /*!< The type of the test */
        std::string name;       /*!< The name of the test */
        u32 world_rank;         /*!< MPI rank that performed the test */
        TestAssertList asserts; /*!< List of the asserts performed withing the test */
        std::string tex_output; /*!< Tex output of the test */

        [[deprecated]]
        TestDataList test_data; /*!< Data returned by the test */

        /**
         * @brief Constructructor
         *
         * @param type
         * @param name
         * @param world_rank
         */
        inline TestResult(const TestType &type, std::string name, const u32 &world_rank)
            : type(type), name(std::move(name)), world_rank(world_rank), asserts{}, test_data() {}

        /**
         * @brief serialize the result of the test
         *
         * @return std::string the serialized results
         */
        std::string serialize_json();

        std::vector<u8> serialize();
    };

} // namespace shamtest::details