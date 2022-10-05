
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
    pymodel_basic_sph_gas.cpp

    pymodel_basic_sph_gas_selfgrav.cpp
    pymodelsetup_sph.cpp
    PUBLIC
        
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
        #patchdata.hpp
        #scheduler.hpp
    )

endif()

if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    pymodel_basic_sph_gas.cpp
    pymodelsetup_sph.cpp
    PUBLIC
        #patchdata.hpp
        #scheduler.hpp
    )


endif()
