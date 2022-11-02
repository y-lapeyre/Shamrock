
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_patchdata_field.cpp
    PUBLIC
         
    )


endif()
