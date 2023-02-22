
if("${BUILD_SIM}")

target_sources(shamrock_lib
    PRIVATE
        sycl_algs.cpp
    PUBLIC
    )

endif()


