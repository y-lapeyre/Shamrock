
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        mpi_handler.cpp
        sycl_handler.cpp
        sycl_mpi_interop.cpp
        cmdopt.cpp
        log.cpp
    PUBLIC
        mpi_handler.hpp
        sycl_handler.hpp
    )

endif()


if("${BUILD_VISU}")

target_sources(shamrock_visu
    PRIVATE
        #mpi_handler.cpp
    PUBLIC
        #mpi_handler.hpp
        sycl_handler.hpp
    )

endif()

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        mpi_handler.cpp
        sycl_handler.cpp
        sycl_mpi_interop.cpp
        cmdopt.cpp
        log.cpp
    PUBLIC
        mpi_handler.hpp
        sycl_handler.hpp
    )


endif()