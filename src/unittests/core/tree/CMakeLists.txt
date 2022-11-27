
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_radix_tree.cpp
    PUBLIC
         
    )


endif()
add_subdirectory(kernels)