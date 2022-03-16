

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
add_subdirectory(sys)
add_subdirectory(scheduler)
add_subdirectory(patch)