
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        patch.cpp
        patchdata.cpp
        patchtree.cpp
        serialpatchtree.cpp
        patchdata_buffer.cpp
    PUBLIC
        patch.hpp
        
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
        patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )

endif()

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        #scheduler.cpp
        patch.cpp
        patchdata.cpp
        patchtree.cpp
        serialpatchtree.cpp
        patchdata_buffer.cpp
    PUBLIC
        patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()