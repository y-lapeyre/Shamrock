cmake_minimum_required(VERSION 3.9)

# TODO rename to shamrock_lib when the old one will be removed

project(Shamlib CXX C)

# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE shamlib_srcs CONFIGURE_DEPENDS "*.cpp")

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamlib target (Shared lib)")
add_library(shamlib SHARED ${shamlib_srcs})
else()
message("-- adding shamlib target (object lib)")
add_library(shamlib OBJECT ${shamlib_srcs})
endif()

add_library(shamlib::shamlib ALIAS shamlib)


target_link_libraries(shamlib PUBLIC shambackends)
target_link_libraries(shamlib PUBLIC shammath)
target_link_libraries(shamlib PUBLIC shamphys)
target_link_libraries(shamlib PUBLIC shamalgs)
target_link_libraries(shamlib PUBLIC shamsys)
target_link_libraries(shamlib PUBLIC shambindings)
target_link_libraries(shamlib PUBLIC shamtree)
target_link_libraries(shamlib PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shamlib PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
