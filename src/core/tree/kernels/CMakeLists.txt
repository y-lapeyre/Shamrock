
if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            karras_alg.cpp
            key_morton_sort.cpp
            morton_kernels.cpp
            reduction_alg.cpp
            compute_ranges.cpp
            convert_ranges.cpp
        PUBLIC
            karras_alg.hpp
            key_morton_sort.hpp
            morton_kernels.hpp
            reduction_alg.hpp
            compute_ranges.hpp
            convert_ranges.hpp
    )


endif()



if("${BUILD_SIM}")

    target_sources(shamrock_main
        PRIVATE
            karras_alg.cpp
            key_morton_sort.cpp
            morton_kernels.cpp
            reduction_alg.cpp
            compute_ranges.cpp
            convert_ranges.cpp
        PUBLIC
            karras_alg.hpp
            key_morton_sort.hpp
            morton_kernels.hpp
            reduction_alg.hpp
            compute_ranges.hpp
            convert_ranges.hpp
    )


endif()
