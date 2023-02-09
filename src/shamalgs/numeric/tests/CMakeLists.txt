
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        numericTests.cpp
    PUBLIC
         
    )


endif()