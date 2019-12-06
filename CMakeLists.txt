###############################################################################
# Vulkan-Interpolator: interpolator/rasterizer based on Vulkan API
# 
# Copyright (c) 2019-2020 Dmitriy Korchemkin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

project(VulkanInterpolator CXX)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Werror)

find_package(Vulkan REQUIRED)
set(BUILD_TESTS_VULKAN_INTERPOLATOR
    TRUE
    CACHE BOOL "Build vulkan interpolator tests")
if(${BUILD_TESTS_VULKAN_INTERPOLATOR} AND NOT TARGET gtest_main)
  find_package(Eigen3 REQUIRED CONFIG)
  if(NOT TARGET gtest_main)
    add_subdirectory(external/googletest)
  endif()
endif()

add_subdirectory(external/delaunator_cpp)
add_subdirectory(vulkan_interpolator)
