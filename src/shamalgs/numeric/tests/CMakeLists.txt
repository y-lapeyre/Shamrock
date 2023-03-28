
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    numericTests.cpp
    exclusiveScanTests.cpp
    numericFallbackTests.cpp
    PUBLIC
         
    )


endif()