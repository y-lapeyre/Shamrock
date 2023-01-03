
if("${BUILD_SIM}")

target_sources(shamrock_main
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