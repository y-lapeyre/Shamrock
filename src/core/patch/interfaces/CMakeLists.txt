
if("${BUILD_SIM}")

target_sources(shamrock_main
    PRIVATE
        #scheduler.cpp
        interface_generator.cpp
        interface_handler.cpp
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        #scheduler.cpp
        interface_generator.cpp
        interface_handler.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()