######################################
# Cmake build configuration for shamrock
######################################

cmake_minimum_required(VERSION 3.9)
project(Shamrock CXX C)


# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

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
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda-sycldevice ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda ")
    endif()

    if("${SyCL_Compiler_BE}" STREQUAL "CPU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice")
    endif()

endif()


######################
# Add compilation flags depending upon the build type
######################
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wunknown-cuda-version -Wall -Wextra -Wno-linker-warnings")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -Wunknown-cuda-version -Wno-linker-warnings")

if (CMAKE_BUILD_TYPE STREQUAL "GCOV")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "COVERAGE_MAP")
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-instr-generate -fcoverage-mapping -fcoverage-compilation-dir=. -fno-sycl-use-footer")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage -fprofile-arcs -fprofile-instr-generate -ftest-coverage -fcoverage-mapping -fno-sycl-use-footer")

endif()

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")


######################
# Add compilation flags for xray if enabled
######################
if("${XRAY_INSTRUMENTATION}")
    message("-- add llvm-xray instrumentation")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fxray-instrument")
endif()

######################
# Add compilation utilisation flags
######################

#morton precision
if("${Morton_precision}" STREQUAL "double")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRECISION_MORTON_DOUBLE")
endif()

#physics precision
if("${Physics_precision}" STREQUAL "single")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRECISION_FULL_SINGLE")
endif()

if("${Physics_precision}" STREQUAL "mixed")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRECISION_MIXED")
endif()

if("${Physics_precision}" STREQUAL "double")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRECISION_FULL_DOUBLE")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lmpi_cxx")


######################
# Add external libraries
######################
set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)









######################
# Add subdirectories
######################

#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp
  )

if("${BUILD_SPH}")

    message(">>> adding sph target <<<")
    add_executable(shamrock_sph main_sph.cpp version.cpp)
    target_link_libraries(shamrock_sph PUBLIC MPI::MPI_C)

endif()

if("${BUILD_AMR}")

    message(">>> adding amr target <<<")
    add_executable(shamrock_amr main_amr.cpp version.cpp)
    target_link_libraries(shamrock_amr PUBLIC MPI::MPI_C)

endif()

if("${BUILD_VISU}")

    message(">>> adding visualisation target <<<")

    add_executable(shamrock_visu main_visu.cpp version.cpp)

endif()

if("${BUILD_TEST}")

    message(">>> adding test target <<<")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test main_test.cpp  version.cpp)
    target_link_libraries(shamrock_test PUBLIC MPI::MPI_C)

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()

add_subdirectory(sys)
add_subdirectory(tree)
add_subdirectory(unittests)
add_subdirectory(scheduler)
