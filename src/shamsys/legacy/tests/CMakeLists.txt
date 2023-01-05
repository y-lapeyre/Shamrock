

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_mpi_handler.cpp
        test_sycl_mpi_interop.cpp
    PUBLIC 
    )


endif()

