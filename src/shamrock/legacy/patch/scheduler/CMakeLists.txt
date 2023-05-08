
if("${BUILD_SIM}")

target_sources(shamrock_lib
    PRIVATE
        #scheduler.cpp
        
        
        
        
        scheduler_mpi.cpp 
        scheduler_patch_list.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



add_subdirectory(tests)