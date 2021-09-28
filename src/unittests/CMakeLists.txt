

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        unittests.cpp 
        unit_test_handler.cpp
    PUBLIC
        unittests.hpp 
        unit_test_handler.hpp
    )


endif()

add_subdirectory(tree)
