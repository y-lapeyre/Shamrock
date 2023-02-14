
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        AMRTmpTests.cpp
    PUBLIC
         
    )


endif()