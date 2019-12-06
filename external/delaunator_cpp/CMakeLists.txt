cmake_minimum_required(VERSION 3.8)
project(delaunator VERSION 0.4.0)
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_library(delaunator src/delaunator.cpp)
target_include_directories(delaunator PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

