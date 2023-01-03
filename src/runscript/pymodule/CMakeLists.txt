
if("${BUILD_SIM}")

target_sources(shamrock_main
    PRIVATE
    pyshamrockcontext.cpp
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()


if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    pyshamrockcontext.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

add_subdirectory(models)
