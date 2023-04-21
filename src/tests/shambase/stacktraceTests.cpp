// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"

#include "shambase/stacktrace.hpp"

TestStart(Unittest, "shambase/stacktrace/print_stack", test_stackprinter, 1){
    StackEntry stack {};

    logger::raw_ln(shambase::fmt_callstack());

    //throw shambase::throw_with_loc<std::invalid_argument>("test");
}