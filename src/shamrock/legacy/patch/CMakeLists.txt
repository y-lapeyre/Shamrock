
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        #patchdata_buffer.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

        
    #patchdata_buffer.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

add_subdirectory(base)
add_subdirectory(scheduler)
add_subdirectory(interfaces)
add_subdirectory(utility)
add_subdirectory(tests)
