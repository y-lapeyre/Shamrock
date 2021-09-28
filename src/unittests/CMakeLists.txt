add_subdirectory(tree)

add_library(unittests unittests.cpp unit_test_handler.cpp)

link_libraries(unittests unittests_tree)