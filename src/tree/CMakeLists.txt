
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        radix_tree.cpp 
        karras_alg.cpp
    PUBLIC
        morton.hpp
        radix_tree.hpp 
        karras_alg.hpp
    )


endif()
