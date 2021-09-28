
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_morton.cpp
        test_karras.cpp
    PUBLIC
        test_tree.hpp 
    )


endif()
