
if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            karras_alg.cpp
            key_morton_sort.cpp
            morton_kernels.cpp
        PUBLIC
            karras_alg.hpp
            key_morton_sort.hpp
            morton_kernels.hpp
    )


endif()
