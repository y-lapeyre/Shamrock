
if("${BUILD_SIM}")

target_sources(shamrock_main
    PRIVATE
    pymodel_basic_sph_gas.cpp

    pymodel_basic_sph_gas_uint.cpp
    pymodelsetup_sph.cpp
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
