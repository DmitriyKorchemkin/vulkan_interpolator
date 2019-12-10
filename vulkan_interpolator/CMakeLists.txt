add_library(vulkan_interpolator src/VulkanInterpolator.cpp)

target_include_directories(
  vulkan_interpolator
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(vulkan_interpolator Vulkan::Vulkan glfw)

set(CMAKE_GLSLC /home/dka/vulkan/1.1.126.0/x86_64/bin/glslc)

function(add_shader_library)
  cmake_parse_arguments(AS_ARGS "" "NAMESPACE;LIBRARY;HEADER_NAME" "SHADERS"
                        ${ARGN})
  message(STATUS "Shader files: ${AS_ARGS_SHADERS}")
  message(STATUS "Namespace: ${AS_ARGS_NAMESPACE}")
  message(STATUS "Header name: ${AS_ARGS_HEADER_NAME}")
  message(STATUS "Library: ${AS_ARGS_LIBRARY}")

  list(APPEND SHADER_LIST)
  foreach(SHADER ${AS_ARGS_SHADERS})
    message(STATUS "Need to compile ${SHADER}")
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
      OUTPUT ${OUTPUT_FILE}
      COMMAND ${CMAKE_GLSLC} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o
              ${OUTPUT_FILE}
      DEPENDS ${SHADER}
      BYPRODUCTS ${OUTPUT_FILE})
    list(APPEND SHADER_LIST ${OUTPUT_FILE})
  endforeach()

  set(HEADER ${CMAKE_CURRENT_BINARY_DIR}/${AS_ARGS_HEADER_NAME})

  add_custom_command(
    OUTPUT ${HEADER}
    COMMAND
      ${CMAKE_CURRENT_SOURCE_DIR}/create_shader_header.sh ARGS --output
      "${HEADER}" --namespace "${AS_ARGS_NAMESPACE}" ${SHADER_LIST}
    DEPENDS ${SHADER_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/create_shader_header.sh)
  target_sources(${AS_ARGS_LIBRARY} PRIVATE ${HEADER})
  target_include_directories(${AS_ARGS_LIBRARY} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

add_shader_library(
  SHADERS
  shaders/shader.vert
  shaders/shader.frag
  HEADER_NAME
  shaders.hpp
  NAMESPACE
  shaders
  LIBRARY
  vulkan_interpolator)
