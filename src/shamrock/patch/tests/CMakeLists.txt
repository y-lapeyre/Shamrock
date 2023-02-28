
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    patchTests.cpp
    CommBufferPatchDataFieldTests.cpp
    CommBufferPatchDataTests.cpp
    PUBLIC
         
    )


endif()