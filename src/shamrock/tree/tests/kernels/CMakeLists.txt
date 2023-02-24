
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        key_pair_sortTests.cpp
    PUBLIC
         
    )


endif()
