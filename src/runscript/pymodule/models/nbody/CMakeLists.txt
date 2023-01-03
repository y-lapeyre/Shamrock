
if("${BUILD_SIM}")

target_sources(shamrock_main
    PRIVATE
    pymodel_nbody_selfgrav.cpp
    pymodel_nbody_setup.cpp
    PUBLIC
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()



if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    pymodel_nbody_selfgrav.cpp
    pymodel_nbody_setup.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()
