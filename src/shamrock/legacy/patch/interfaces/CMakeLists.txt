
if("${BUILD_SIM}")

target_sources(shamrock_lib
    PRIVATE
        #scheduler.cpp
        interface_generator.cpp
        interface_handler.cpp
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()

