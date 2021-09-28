#include "unittests.hpp"

#include "tree/test_tree.hpp"

void run_tree_tests(){
    run_tests_morton();
    run_tests_karras_alg();
}

void run_tests(){
    run_tree_tests();
}