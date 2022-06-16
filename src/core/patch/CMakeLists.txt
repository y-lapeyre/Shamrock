
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        
        global_var.cpp
        loadbalancing_hilbert.cpp 
        patch.cpp 
        patchdata_buffer.cpp 
        patchdata.cpp 
        patchtree.cpp 
        scheduler_mpi.cpp 
        scheduler_patch_data.cpp 
        scheduler_patch_list.cpp 
        serialpatchtree.cpp


        
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

        
        
    global_var.cpp
    loadbalancing_hilbert.cpp 
    patch.cpp 
    patchdata_buffer.cpp 
    patchdata.cpp 
    patchtree.cpp 
    scheduler_mpi.cpp 
    scheduler_patch_data.cpp 
    scheduler_patch_list.cpp 
    serialpatchtree.cpp


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

add_subdirectory(interfaces)