
if("${BUILD_SPH}")

target_sources(shamrock_sph
    PRIVATE
        mpi_handler.cpp
    PUBLIC
        mpi_handler.hpp
        sycl_handler.hpp
    )

endif()

if("${BUILD_AMR}")

target_sources(shamrock_amr
    PRIVATE
        mpi_handler.cpp
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
    PUBLIC
        mpi_handler.hpp
        sycl_handler.hpp
    )


endif()