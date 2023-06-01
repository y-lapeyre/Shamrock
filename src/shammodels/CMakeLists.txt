target_sources(shamrock_lib
    PRIVATE
        BasicGas.cpp 
        BasicSPHGhosts.cpp
        SPHSolverImpl.cpp
        SPHModel.cpp
    PUBLIC
    )



#add_subdirectory(algs)
#add_subdirectory(physics)
#add_subdirectory(setup)
add_subdirectory(generic)
add_subdirectory(sph)
add_subdirectory(nbody)
add_subdirectory(bindings)
add_subdirectory(amr)