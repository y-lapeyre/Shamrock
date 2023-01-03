
if("${BUILD_SIM}")

target_sources(shamrock_main
    PRIVATE
        sycl_algs.cpp
    PUBLIC
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        sycl_algs.cpp
    PUBLIC
    )


endif()