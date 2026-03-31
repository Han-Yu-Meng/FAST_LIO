cmake_minimum_required(VERSION 3.10)
project(livox_msg)

add_library(livox_msg INTERFACE)

target_include_directories(livox_msg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
