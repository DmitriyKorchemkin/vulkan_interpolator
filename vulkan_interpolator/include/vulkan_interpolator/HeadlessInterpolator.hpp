/******************************************************************************
 * Vulkan-Interpolator: interpolator/rasterizer based on Vulkan API
 * 
 * Copyright (c) 2019-2020 Dmitriy Korchemkin
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/
#ifndef VULKAN_INTERPOLATOR_INTERPOLATOR_HEADLESS_HPP
#define VULKAN_INTERPOLATOR_INTERPOLATOR_HEADLESS_HPP

#include <mutex>
#include <random>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "vulkan_interpolator/Interpolator.hpp"

namespace vulkan_interpolator {

struct HeadlessInterpolator {
  HeadlessInterpolator(
      const std::vector<size_t> &devices,
      const InterpolationOptions &opts = InterpolationOptions(),
      vk::Instance *sharedInstance = nullptr);
  ~HeadlessInterpolator();
  // Triangulate and rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int width, const int height, const int stride_bytes,
                   float *output);
  // Just rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int nTriangles, const int *indicies, const int width,
                   const int height, const int stride_bytes, float *output);

  // Compute delaunay triangulation
  static void PrepareInterpolation(const int nPoints, const float *points,
                                   std::vector<int> &indicies);

private:
  std::mutex mutex;
  // allocate new vertex data buffer if needed
  bool allocatePoints();
  // allocate new index data buffer if needed
  bool allocateIndicies();
  // allocate output image if needed, setup scissors, scaler
  bool allocateImage();
  // stage vertex data, setup copying
  void setupVertices();
  // setup copying output image
  void setupCopyImage();
  // setup renderpass
  void setupRenderpass();
  // Perform rasterizer run & map image memory
  void rasterize();

  uint32_t heightAllocated = 0, widthAllocated = 0, pointsAllocated = 0,
           indiciesAllocated = 0, widthLast = -1, heightLast = -1;
  size_t stagingAllocated = 0;

  uint32_t height, width, points, indicies;
  // points into staging buffer
  float *points_ptr;
  // points into staging buffer
  int32_t *indicies_ptr;

  const vk::Format format1d = vk::Format::eR32Sfloat,
                   format2d = vk::Format::eR32G32Sfloat;
  vk::UniqueInstance instance;
  vk::Instance *sharedInstance = nullptr;
  vk::UniqueDebugUtilsMessengerEXT messenger;
  vk::UniqueDevice device;
  vk::UniqueShaderModule vertexShaderModule, fragmentShaderModule;
  vk::UniquePipelineLayout pipelineLayout;
  vk::UniqueRenderPass renderPass;
  vk::UniquePipeline pipeline;
  vk::UniqueFramebuffer framebuffer;
  vk::UniqueCommandPool commandPoolUnique;
  vk::Queue deviceQueue;
  vk::UniqueCommandBuffer copyBuffer, copyBackBuffer, renderBuffer;
  vk::PhysicalDevice physicalDevice;
  using BufferMem = std::pair<vk::UniqueDeviceMemory, vk::UniqueBuffer>;
  BufferMem vertexBuffer, indexBuffer, stagingBuffer;
  vk::UniqueDeviceMemory outputMem, renderMem;
  vk::UniqueImage image;
  vk::UniqueImageView imageView;
  vk::UniqueImage outputImage;
  void *stagingData;
  vk::UniqueFence fence;
  vk::ClearValue clearValues;

  BufferMem createBuffer(size_t sz, const vk::BufferUsageFlags &usage,
                         const vk::MemoryPropertyFlags &flags);

  uint32_t findMemoryType(uint32_t mask,
                          const vk::MemoryPropertyFlags &properties);
};

} // namespace vulkan_interpolator

#endif
