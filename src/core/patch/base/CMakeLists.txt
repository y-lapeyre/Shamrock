
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        
        patch.cpp 
        patchdata.cpp 
        patchdata_field.cpp 
        patchdata_layout.cpp 
        patchtree.cpp 


        
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

    patch.cpp 
    patchdata.cpp 
    patchdata_field.cpp 
    patchdata_layout.cpp 
    patchtree.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()