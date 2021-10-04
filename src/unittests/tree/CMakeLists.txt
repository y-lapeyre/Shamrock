
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_morton.cpp
        test_karras.cpp
        test_key_pair_sort.cpp
    PUBLIC
        test_tree.hpp 
    )


endif()
