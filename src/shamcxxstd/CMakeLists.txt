cmake_minimum_required(VERSION 3.9)

project(Shamcxxstd CXX C)

message("-- adding shamcxxstd target")
add_library(shamcxxstd INTERFACE)
add_library(shamcxxstd::shamcxxstd ALIAS shamcxxstd)


target_include_directories(shamcxxstd INTERFACE
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")

