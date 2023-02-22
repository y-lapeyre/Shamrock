
set(Sources
    mpi_handler.cpp
    sycl_handler.cpp
    sycl_mpi_interop.cpp
    cmdopt.cpp
    log.cpp
)
set(SourcesTest
    
)
set(Headers
    mpi_handler.hpp
    sycl_handler.hpp
)
set(HeadersTest

)

if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
     ${SourcesTest}
    PUBLIC
     ${HeadersTest}
    )
endif()


add_subdirectory(tests)
