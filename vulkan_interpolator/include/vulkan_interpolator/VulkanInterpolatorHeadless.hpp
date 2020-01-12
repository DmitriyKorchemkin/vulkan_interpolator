#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <mutex>
#include <random>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vulkan_interpolator {

struct InterpolationOptions {
  int heightPreallocated = 1000;
  int widthPreallocated = 1000;
  int pointsPreallocated = 10000;
  int indiciesPreallocated = 30000;
};

struct HeadlessInterpolator {
  HeadlessInterpolator(
      const std::vector<size_t> &devices,
      const InterpolationOptions &opts = InterpolationOptions());
  ~HeadlessInterpolator();
  // Triangulate and rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int width, const int height, const int stride_bytes,
                   float *output, float dt, float db, float dl, float dr,
                   float sx, float sy);
  // Just rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int nTriangles, const int *indicies, const int width,
                   const int height, const int stride_bytes, float *output,
                   float dt, float db, float dl, float dr, float sx, float sy);

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
