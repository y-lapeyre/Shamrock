
if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            radix_tree.cpp 
        PUBLIC
            morton.hpp
            radix_tree.hpp 
    )


endif()

add_subdirectory(kernels)