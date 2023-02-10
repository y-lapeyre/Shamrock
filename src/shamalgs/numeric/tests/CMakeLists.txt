
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    numericTests.cpp
    exclusiveScanGPUGems39Tests.cpp
    numericFallbackTests.cpp
    PUBLIC
         
    )


endif()