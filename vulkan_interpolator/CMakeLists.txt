add_library(vulkan_interpolator src/VulkanInterpolator.cpp)

target_include_directories(vulkan_interpolator PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(vulkan_interpolator Vulkan::Vulkan)


