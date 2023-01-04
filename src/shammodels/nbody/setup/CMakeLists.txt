
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        nbody_setup.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()


if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

        
        
    nbody_setup.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()
