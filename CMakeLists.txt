project(VulkanInterpolator CXX)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Werror)# -fsanitize=address)
#link_libraries("-fsanitize=address")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(external/delaunator_cpp)
add_subdirectory(vulkan_interpolator)

add_subdirectory(tests)
