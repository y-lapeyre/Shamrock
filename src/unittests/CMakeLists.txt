

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        shamrocktest.cpp
        test_random.cpp
    PUBLIC
        shamrocktest.hpp 
    )


endif()

add_subdirectory(tree)
