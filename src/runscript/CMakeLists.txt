
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
    rscripthandler.cpp
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()


if("${BUILD_VISU}")

target_sources(shamrock_visu
    PRIVATE
        #scheduler.cpp
        #patch.cpp
        #patchdata.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )

endif()

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    rscripthandler.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()