#pragma once

#include "tree/test_morton.hpp"
#include "tree/test_karras.hpp"

inline void add_tests_global(){

    add_test("string_utils.hpp/format()",false, []{

        UTest_NOMPI_assert(R"=(format("%s %d\n","ttt : ",10))=", ! format("%s %d\n","ttt : ",10).compare("ttt :  10\n"));

    });


    add_tests_morton();

    add_tests_karras_alg();

}