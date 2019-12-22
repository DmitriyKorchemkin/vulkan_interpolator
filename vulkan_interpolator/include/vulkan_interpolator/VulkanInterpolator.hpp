#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <vector>
#include <random>
#include <vulkan/vulkan.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vulkan_interpolator {

struct Interpolator {
    static const uint32_t width = 1536;
    static const uint32_t height = 1536;
    GLFWwindow* window;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::UniqueSwapchainKHR swapChain;
    std::vector<vk::UniqueImageView> imageViews;
    vk::UniqueShaderModule vertexShaderModule, fragmentShaderModule;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipeline pipeline;
    std::vector<vk::UniqueFramebuffer> framebuffers;
    vk::UniqueCommandPool commandPoolUnique;
    vk::Queue presentQueue, deviceQueue;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    vk::UniqueCommandBuffer copyBuffer;
   vk::PhysicalDevice physicalDevice;
   using BufferMem = 
   std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>;
  BufferMem vertexBuffer, indexBuffer, stagingBuffer;
  void* stagingData;
    

  Interpolator(const std::vector<size_t> &devices);
  ~Interpolator();
  void run();

private:
  const int N_PTS = 10000;
  const int PTS_SIZE = 6 * N_PTS * sizeof(float);
  int nTris = 0;
  const size_t IDX_MAX_CNT = 30*N_PTS;
  const size_t IDX_MAX_SIZE = IDX_MAX_CNT* sizeof(uint32_t);
  
  std::vector<float> points;
  std::vector<int> indicies;
  std::mt19937 rng;

  void createPoints();

  BufferMem
  createBuffer(size_t sz, const vk::BufferUsageFlags &usage, const  vk::MemoryPropertyFlags &flags);
  void createVertexBuffer();
  void createIndexBuffer();
  void createStagingBuffer();
  void stageData();

  uint32_t findMemoryType(uint32_t mask,  const vk::MemoryPropertyFlags &properties);

  std::vector<const char *> required_extensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
};

} // namespace vulkan_interpolator

#endif
