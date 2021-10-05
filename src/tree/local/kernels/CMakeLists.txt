
if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            karras_alg.cpp
            key_morton_sort.cpp
            morton_kernels.cpp
            reduction_alg.cpp
        PUBLIC
            karras_alg.hpp
            key_morton_sort.hpp
            morton_kernels.hpp
            reduction_alg.hpp
    )


endif()
