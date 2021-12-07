#include "unittests.hpp"

#include "tree/test_tree.hpp"

#include "scheduler/test_scheduler.hpp"

void run_tree_tests(){
    run_tests_morton();
    run_tests_karras_alg();
    run_tests_morton_code_sort();
    run_tests_scheduler();
}

void run_tests(){
    run_tree_tests();
}