
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    tree_performance.cpp
    PUBLIC
         
    )


endif()