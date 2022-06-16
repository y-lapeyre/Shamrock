
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        scheduler_mpi.cpp
        scheduler_patch_list.cpp
        scheduler_patch_data.cpp
        loadbalancing_hilbert.cpp
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
        #scheduler.cpp
        scheduler_mpi.cpp
        scheduler_patch_list.cpp
        scheduler_patch_data.cpp
        loadbalancing_hilbert.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()