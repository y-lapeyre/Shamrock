
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_radix_tree.cpp

        test_karras.cpp
        test_key_pair_sort.cpp 
        test_morton.cpp
    PUBLIC
         
    )


endif()
add_subdirectory(kernels)