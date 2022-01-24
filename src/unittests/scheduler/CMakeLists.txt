
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_mpi_scheduler.cpp
    PUBLIC
         
    )


endif()
