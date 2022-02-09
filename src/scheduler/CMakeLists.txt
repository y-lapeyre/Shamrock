
if("${BUILD_SPH}")

target_sources(shamrock_sph
    PRIVATE
        #scheduler.cpp
        patch.cpp
        patchdata.cpp
        scheduler_mpi.cpp
        scheduler_patch_list.cpp
        scheduler_patch_data.cpp
    PUBLIC
        patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()

if("${BUILD_AMR}")

target_sources(shamrock_amr
    PRIVATE
        #scheduler.cpp
        patch.cpp
        patchdata.cpp
        scheduler_mpi.cpp
        scheduler_patch_list.cpp
        scheduler_patch_data.cpp
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
        scheduler_mpi.cpp
        scheduler_patch_list.cpp
        scheduler_patch_data.cpp
    PUBLIC
        patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()