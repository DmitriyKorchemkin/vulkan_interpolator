#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <random>
#include <vector>
#include <mutex>
#include <vulkan/vulkan.hpp>

namespace vulkan_interpolator {

struct InterpolationOptions {
  int heightPreallocated = 1024;
  int widthPreallocated = 1024;
  int pointsPreallocated = 10000;
  int indiciesPreallocated = 30000;
};

struct HeadlessInterpolator {
  HeadlessInterpolator(const std::vector<size_t> &devices);
  ~HeadlessInterpolator();
  void run();


  // Triangulate and rasterize
  void interpolate(
      const int nPoints,
      const float* points,
      const float* values,
      const int width,
      const int height,
      const int stride_bytes,
      float* output);
  // Just rasterize
  void interpolate(
      const int nPoints,
      const float* points,
      const float* values,
      const int nTriangles,
      const int* indicies,
      const int width,
      const int height,
      const int stride_bytes,
      float* output);


  // Compute delaunay triangulation
  static void PrepareInterpolation(
      const int nPoints,
      const float* points,
      std::vector<int> &indicies);
private:
  std::mutex mutex;
  bool allocatePoints();
  bool allocateIndicies();
  bool allocateImage();
  void setupVertices();
  void setupCopyImage();
  


  int heightAllocated = 0, widthAllocated = 0,  pointsAllocated = 0, indiciesAllocated = 0;

  int height, width, points, indicies;





  const int N_PTS = 10000;
  const int PTS_SIZE = 6 * N_PTS * sizeof(float);
  int nTris = 0;
  const size_t IDX_MAX_CNT = 30 * N_PTS;
  const size_t IDX_MAX_SIZE = IDX_MAX_CNT * sizeof(uint32_t);
  const vk::Format format1d = vk::Format::eR32Sfloat,
        format2d = vk::Format::eR32G32Sfloat;
  static const uint32_t width = 4096;
  static const uint32_t height = 4096;
  vk::UniqueInstance instance;
  vk::UniqueDebugUtilsMessengerEXT messenger;
  vk::UniqueDevice device;
  vk::UniqueImage image;
  vk::UniqueImageView imageView;
  vk::UniqueShaderModule vertexShaderModule, fragmentShaderModule;
  vk::UniquePipelineLayout pipelineLayout;
  vk::UniqueRenderPass renderPass;
  vk::UniquePipeline pipeline;
  vk::UniqueFramebuffer framebuffer;
  vk::UniqueCommandPool commandPoolUnique;
  vk::Queue deviceQueue;
  vk::UniqueCommandBuffer copyBuffer, copyBackBuffer, renderBuffer;
  vk::PhysicalDevice physicalDevice;
  using BufferMem = std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>;
  BufferMem vertexBuffer, indexBuffer, stagingBuffer;
  void *stagingData;
  vk::UniqueImage outputImage;
  vk::UniqueDeviceMemory outputMem, renderMem;
  vk::UniqueFence fence;
  vk::ClearValue clearValues;


  std::vector<float> points;
  std::vector<int> indicies;
  std::mt19937 rng;

  void createPoints();

  BufferMem createBuffer(size_t sz, const vk::BufferUsageFlags &usage,
                         const vk::MemoryPropertyFlags &flags);
  void createVertexBuffer();
  void createIndexBuffer();
  void createStagingBuffer();
  void stageData();

  uint32_t findMemoryType(uint32_t mask,
                          const vk::MemoryPropertyFlags &properties);
};

} // namespace vulkan_interpolator

#endif
