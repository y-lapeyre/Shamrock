
if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            morton.cpp
            radix_tree.cpp 
            karras_alg.cpp
            key_morton_sort.cpp
        PUBLIC
            morton.hpp
            radix_tree.hpp 
            karras_alg.hpp
            key_morton_sort.hpp
    )


endif()
