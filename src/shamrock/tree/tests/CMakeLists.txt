
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        radix_treeTests.cpp
        karrasTests.cpp
        key_pair_sortTests.cpp 
        mortonTests.cpp
    PUBLIC
         
    )


endif()
add_subdirectory(kernels)