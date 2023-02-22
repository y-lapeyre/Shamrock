
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        reductionTests.cpp
    PUBLIC
         
    )


endif()