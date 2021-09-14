######################################
# Cmake build configuration for shamrock
######################################

cmake_minimum_required(VERSION 3.9)
project(Shamrock CXX)


# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)


# Necessary for clangd to work properly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



message(STATUS "chosen compiler : ${SyCL_Compiler}")

if("${SyCL_Compiler}" STREQUAL "DPC++")

    message(STATUS "chosen backend  : ${SyCL_Compiler_BE}")
    message(STATUS "compiler folder : ${DPCPP_PACKAGE_ROOT_DIR}")

    LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${DPCPP_INSTALL_LOC}/bin")
    LIST(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${DPCPP_INSTALL_LOC}/bin/lib")

    set(CMAKE_CXX_COMPILER "${DPCPP_INSTALL_LOC}/bin/clang++")
    set(CMAKE_CXX_FLAGS "-fsycl -std=c++17 -pedantic-errors -fcolor-diagnostics")

    ######################
    # Chose sycl backend
    ######################
    if("${SyCL_Compiler_BE}" STREQUAL "CUDA")
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda-sycldevice ")
    endif()

    if("${SyCL_Compiler_BE}" STREQUAL "CPU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice")
    endif()

endif()


######################
# Add compilation flags depending upon the build type
######################
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wunknown-cuda-version -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -Wunknown-cuda-version")

if (CMAKE_BUILD_TYPE STREQUAL "GCOV")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "COVERAGE_MAP")
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-instr-generate -fcoverage-mapping -fcoverage-compilation-dir=. -fno-sycl-use-footer")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage -fprofile-arcs -fprofile-instr-generate -ftest-coverage -fcoverage-mapping -fno-sycl-use-footer")

endif()




######################
# Add external libraries
######################
set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)





######################
# Add subdirectories
######################
#add_subdirectory(sys)




# if("${BUILD_SPH}")

#     message(">>> adding sph target <<<")

#     add_executable(sphive_sph main_sph.cpp)
#     target_link_libraries(sphive_sph PUBLIC sys)
#     target_link_libraries(sphive_sph PUBLIC MPI::MPI_C)

# endif()


# if("${BUILD_TEST}")

#     message(">>> adding test target <<<")

#     add_executable(sphive_test main_test.cpp)
#     target_link_libraries(sphive_test PUBLIC sys)
#     target_link_libraries(sphive_test PUBLIC MPI::MPI_C)
#     target_link_libraries(sphive_test PUBLIC cunit)

# endif()





# if("${BUILD_VISU}")

#     message(">>> adding visualisation target <<<")

#     add_executable(sphive_visu main_visu.cpp)

# endif()