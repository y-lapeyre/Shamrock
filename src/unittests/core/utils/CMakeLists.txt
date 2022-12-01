
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_sycl_algs.cpp
    PUBLIC
         
    )


endif()