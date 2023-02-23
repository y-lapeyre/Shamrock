
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    patchTests.cpp
    PUBLIC
         
    )


endif()