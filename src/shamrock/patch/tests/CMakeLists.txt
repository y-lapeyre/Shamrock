
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_patch.cpp
    PUBLIC
         
    )


endif()