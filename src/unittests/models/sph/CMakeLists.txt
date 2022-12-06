
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    benchmark_periodic_box.cpp
    benchmark_soundwave.cpp
    PUBLIC
         
    )


endif()
