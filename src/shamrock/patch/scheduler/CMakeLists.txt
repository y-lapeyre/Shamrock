
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        
        loadbalancing_hilbert.cpp 
        scheduler_mpi.cpp 
        scheduler_patch_data.cpp 
        scheduler_patch_list.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

        
        
    loadbalancing_hilbert.cpp 
    scheduler_mpi.cpp 
    scheduler_patch_data.cpp 
    scheduler_patch_list.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

add_subdirectory(tests)