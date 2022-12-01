
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        
    PUBLIC
    )

endif()


if("${BUILD_VISU}")

target_sources(shamrock_visu
    PRIVATE
    PUBLIC
    )

endif()

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        
    PUBLIC
    )


endif()