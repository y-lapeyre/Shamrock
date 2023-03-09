// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "Test.hpp"

#include "shamsys/NodeInstance.hpp"
#include "shamutils/throwUtils.hpp"

namespace shamtest::details {

    TestResult current_test{Unittest, "", -1};

    TestResult Test::run() {

        using namespace shamsys::instance;

        if (node_count != -1) {
            if (node_count != world_size) {
                throw shamutils::throw_with_loc<std::runtime_error>(
                    "trying to run a test with wrong number of nodes"
                );
            }
        }

        current_test = TestResult{type, name, world_rank};

        test_functor();

        return std::move(current_test);
    }

} // namespace shamtest::details